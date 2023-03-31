import os
import sys
import logging
import numpy as np
import math
import configparser
# from ctypes import *
sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer, GLib
import pyds
from app.utils.bus_call import bus_call
from app.utils.is_aarch_64 import is_aarch64
from collections import defaultdict
from functools import partial


class Pipeline:

    def __init__(self, args, urls):
        self.args = args
        self.logger = logging.getLogger(__name__ + "." +
                                        self.__class__.__name__)
        self.video_uri = urls
        self.num_sources = len(self.video_uri)
        self.output_video_path = os.path.join(args.output_path, "out.mp4")
        self.source_enable = [False] * self.num_sources
        self.tracker_config_path = "configs/config_tracker.txt"

        GObject.threads_init()
        Gst.init(None)

        self.logger.info("Creating Pipeline")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            self.logger.error("Failed to create Pipeline")
        # self.source_id_list = [0] * self.num_sources
        self.elements = []
        self.source_bin = None
        self.streammux = None
        self.nvvidconv1 = None
        self.capsfilter1 = None
        self.nvosd = None
        self.tiler = None
        self.nvvidconv2 = None
        self.queue1 = None
        self.sink_bin = None
        self.source_bin_list = [None] * self.num_sources
        self.rtsp_bin_list = [None] * self.num_sources
        self.nvconv = [None]*self.num_sources
        self.streamdemux = None
        self.sinkpad = []
        self._create_elements()
        self._link_elements()

    def __str__(self):
        return " -> ".join([elm.name for elm in self.elements])

    def _add_element(self, element, idx=None):
        if idx:
            self.elements.insert(idx, element)
        else:
            self.elements.append(element)

        self.pipeline.add(element)

    def _create_element(self,
                        factory_name,
                        name,
                        print_name,
                        detail="",
                        add=True):
        """Creates an element with Gst Element Factory make.

        Return the element if successfully created, otherwise print to stderr and return None.
        """
        self.logger.info(f"Creating {print_name}")
        elm = Gst.ElementFactory.make(factory_name, name)

        if not elm:
            self.logger.error(f"Unable to create {print_name}")
            if detail:
                self.logger.error(detail)

        if add:
            self._add_element(elm)

        return elm

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        # print(name)
        if (name.find("decodebin") != -1):
            Object.connect("child-added", self.decodebin_child_added,
                           user_data)
        if (name.find("nvv4l2decoder") != -1):
            if (is_aarch64()):
                print('###########################1')
                Object.set_property("enable-max-performance", True)
                Object.set_property("drop-frame-interval", 0)
                Object.set_property("num-extra-surfaces", 0)
            else:
                Object.set_property("gpu_id", 0)

    def cb_newpad(self, decodebin, pad, data):
        caps = pad.get_current_caps()
        if not caps:
            caps = pad.query_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()

        if (gstname.find("video") != -1):
            source_id = data
            pad_name = "sink_%u" % source_id
            # Get a sink pad from the streammux, link to decodebin
            sinkpad = self.streammux.get_request_pad(pad_name)
            self.sinkpad.append(sinkpad)
            if not pad.link(sinkpad) == Gst.PadLinkReturn.OK:
                sys.stderr.write("Failed to link decodebin to pipeline\n")

    def _create_source_bin(self, index, filename):
        # Create a source GstBin to abstract this bin's content from the rest of the
        # self.source_id_list[index] = index
        bin_name = "source-bin-%02d" % index
        print("Creating {} for {}".format(bin_name, filename))
        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        bin = Gst.ElementFactory.make("nvurisrcbin", bin_name)
        if not bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        # We set the input uri to the source element
        bin.set_property("uri", filename)
        bin.set_property("file-loop", 1)
        bin.set_property("cudadec-memtype", 0)
        bin.set_property("drop-frame-interval", 0)
        bin.set_property("rtsp-reconnect-interval", 10)
        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has been created by the decodebin
        bin.connect("child-added", self.decodebin_child_added, index)
        bin.connect("pad-added", self.cb_newpad, index)
        self.pipeline.add(bin)

        return bin


    def _create_streammux(self):
        streammux = self._create_element("nvstreammux", "stream-muxer",
                                         "Stream mux")
        streammux.set_property('width', self.args.input_width)
        streammux.set_property('height', self.args.input_height)
        streammux.set_property('batch-size', self.num_sources)
        streammux.set_property('batched-push-timeout', 20)
        streammux.set_property('live-source', 1)
        streammux.set_property('sync-inputs', 0)

        return streammux

    def _create_tiler(self):
        tiler = self._create_element("nvmultistreamtiler", "nvtiler", "Tiler")
        tiler_rows = int(math.sqrt(self.num_sources))
        tiler_columns = int(math.ceil((1.0 * self.num_sources) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        tiler.set_property("width", self.args.output_width)
        tiler.set_property("height", self.args.output_height)
        return tiler


    def _create_rtsp_sink_bin(self, port):
        rtsp_sink_bin = Gst.Bin.new("rtsp-sink-bin" + str(port))

        nvvidconv4 = self._create_element("nvvideoconvert",
                                          "convertor3" + str(port),
                                          "Converter 3" + str(port),
                                          add=False)
        # nvosd = self._create_element("nvdsosd",
        #                              "onscreendisplay" + str(port),
        #                              "OSD" + str(port),
        #                              add=False)

        nvvidconv3 = self._create_element("nvvideoconvert",
                                          "convertor" + str(port),
                                          "Converter" + str(port),
                                          add=False)
        capsfilter2 = self._create_element("capsfilter",
                                           "capsfilter" + str(port),
                                           "Caps filter" + str(port),
                                           add=False)
        capsfilter2.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

        if self.args.rtsp_codec not in ["H264", "H265"]:
            raise ValueError(f"Invalid codec '{self.args.rtsp_codec}'")

        # Make the encoder
        encoder = self._create_element(
            f"nvv4l2{self.args.rtsp_codec.lower()}enc",
            "encoder",
            f"{self.args.rtsp_codec} encoder",
            add=False)
        encoder.set_property('bitrate', 4000000)

        if is_aarch64():
            encoder.set_property('preset-level', 1)
            encoder.set_property('insert-sps-pps', 1)
            encoder.set_property('bufapi-version', 1)

        # Make the payload-encode video into RTP packets
        rtppay = self._create_element(f"rtp{self.args.rtsp_codec.lower()}pay",
                                      "rtppay" + str(port),
                                      f"{self.args.rtsp_codec} rtppay" +
                                      str(port),
                                      add=False)

        # Make the UDP sink
        # updsink_port = self.args.port
        sink = self._create_element("udpsink",
                                    "udpsink" + str(port),
                                    "UDP sink" + str(port),
                                    add=False)
        sink.set_property('host', '224.224.255.255')
        sink.set_property('port', port)
        sink.set_property('async', False)
        sink.set_property('sync', self.args.sync)
        sink.set_property("qos", 0)

        rtsp_sink_bin.add(nvvidconv4)
        # rtsp_sink_bin.add(nvosd)
        rtsp_sink_bin.add(nvvidconv3)
        rtsp_sink_bin.add(capsfilter2)
        rtsp_sink_bin.add(encoder)
        rtsp_sink_bin.add(rtppay)
        rtsp_sink_bin.add(sink)

        rtsp_sink_bin.add_pad(
            Gst.GhostPad.new("sink", nvvidconv4.get_static_pad("sink")))
        self._link_sequential([
            nvvidconv4, nvvidconv3, capsfilter2, encoder, rtppay, sink
        ])
        self._add_element(rtsp_sink_bin)

        return rtsp_sink_bin

    def _create_tracker(self):
        tracker = self._create_element("nvtracker", "tracker", "Tracker")

        config = configparser.ConfigParser()
        config.read(self.tracker_config_path)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width':
                tracker_width = config.getint('tracker', key)
                tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height':
                tracker_height = config.getint('tracker', key)
                tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id':
                tracker_gpu_id = config.getint('tracker', key)
                tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file':
                tracker_ll_lib_file = config.get('tracker', key)
                tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file':
                tracker_ll_config_file = config.get('tracker', key)
                tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process':
                tracker_enable_batch_process = config.getint('tracker', key)
                tracker.set_property('enable_batch_process',
                                     tracker_enable_batch_process)
            if key == 'enable-past-frame':
                tracker_enable_past_frame = config.getint('tracker', key)
                tracker.set_property('enable_past_frame',
                                     tracker_enable_past_frame)
            if key == 'display-tracking-id':
                tracker_enable_display_id = config.getint('tracker', key)
                tracker.set_property('display-tracking-id',
                                     tracker_enable_display_id)

        return tracker

    def _create_elements(self):
        self.streammux = self._create_streammux()

        for i in range(self.num_sources):
            source_bin = self._create_source_bin(i, self.video_uri[i])
            self.source_bin_list[i] = source_bin


        if self.args.tiler:
            self.tiler = self._create_tiler()
            #Use convertor to convert from NV12 to RGBA (easier to work with in Python)

            if self.args.output_format.lower() == "rtsp":
                self.sink_bin = self._create_rtsp_sink_bin(self.args.port)

            if not is_aarch64():
                # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
                mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
                self.nvvidconv1.set_property("nvbuf-memory-type", mem_type)
                self.tiler.set_property("nvbuf-memory-type", mem_type)
                self.nvvidconv2.set_property("nvbuf-memory-type", mem_type)

        else:
            self.streamdemux = self._create_element(
                "nvstreamdemux", "Stream-demuxer", "Stream-demuxer")
            for i in range(self.num_sources):
                srcpad1 = self.streamdemux.get_request_pad("src_%u" % i)
                if not srcpad1:
                    sys.stderr.write(
                        " Unable to get the src pad of streamdemux \n")
                nvvidconv = self._create_element(
                    "nvvideoconvert", "convertor" + str(i), "Converter " + str(i), add=False)
                capsfilter = self._create_element(
                    "capsfilter", "capsfilter " + str(i), "Caps filter " + str(i), add=False)

                if not is_aarch64():
                    # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
                    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
                    nvvidconv.set_property("nvbuf-memory-type", mem_type)

                self.pipeline.add(nvvidconv)
                self.pipeline.add(capsfilter)
                capsfilter.set_property("caps",
                                        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
                sink_bin = self._create_rtsp_sink_bin(self.args.port - i - 1)
                # self.pipeline.add(sink_bin)
                self.rtsp_bin_list[i] = sink_bin

                sinkpad1 = nvvidconv.get_static_pad("sink")
                srcpad1.link(sinkpad1)
                nvvidconv.link(capsfilter)
                capsfilter.link(sink_bin)
                self.nvconv[i] = capsfilter
            if not is_aarch64():
                # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
                mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
                # self.nvvidconv1.set_property("nvbuf-memory-type", mem_type)

    def add_source(self, index, filename):
        source_bin = self._create_source_bin(index, filename)
        self.source_bin_list[index] = source_bin
        state_return = source_bin.set_state(Gst.State.PLAYING)

        if state_return == Gst.StateChangeReturn.SUCCESS:
            print("STATE CHANGE SUCCESS\n")

        elif state_return == Gst.StateChangeReturn.FAILURE:
            print("STATE CHANGE FAILURE\n")

        elif state_return == Gst.StateChangeReturn.ASYNC:
            state_return = self.source_bin_list[index].get_state(
                Gst.CLOCK_TIME_NONE)

        elif state_return == Gst.StateChangeReturn.NO_PREROLL:
            print("STATE CHANGE NO PREROLL\n")

    def stop_release_source(self, source_id):
        #Attempt to change status of source to be released
        state_return = self.source_bin_list[source_id].set_state(
            Gst.State.NULL)

        if state_return == Gst.StateChangeReturn.SUCCESS:
            print("STATE CHANGE SUCCESS\n")
            pad_name = "sink_%u" % source_id
            print(pad_name)
            #Retrieve sink pad to be released
            sinkpad = self.streammux.get_static_pad(pad_name)
            #Send flush stop event to the sink pad, then release from the streammux
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            self.streammux.release_request_pad(sinkpad)

            #Remove the source bin from the pipeline
            self.pipeline.remove(self.source_bin_list[source_id])
            self.source_bin_list[source_id] = None
            print("STATE CHANGE SUCCESS\n")

        elif state_return == Gst.StateChangeReturn.FAILURE:
            print("STATE CHANGE FAILURE\n")

        elif state_return == Gst.StateChangeReturn.ASYNC:
            state_return = self.source_bin_list[source_id].get_state(
                Gst.CLOCK_TIME_NONE)
            pad_name = "sink_%u" % source_id
            print(pad_name)
            sinkpad = self.streammux.get_static_pad(pad_name)
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            self.streammux.release_request_pad(sinkpad)
            print("STATE CHANGE ASYNC\n")
            self.pipeline.remove(self.source_bin_list[source_id])
            self.source_bin_list[source_id] = None

    @staticmethod
    def _link_sequential(elements: list):
        for i in range(0, len(elements) - 1):
            elements[i].link(elements[i + 1])

    def _link_elements(self):
        self.logger.info(f"Linking elements in the Pipeline: {self}")
        self._link_sequential(self.elements)

    @staticmethod
    def _get_static_pad(element, pad_name: str = "sink"):
        pad = element.get_static_pad(pad_name)
        if not pad:
            raise AttributeError(
                f"Unable to get {pad_name} pad of {element.name}")

        return pad

    def release(self):
        """Release resources and cleanup."""
        pass

    def run(self):
        # Create an event loop and feed gstreamer bus messages to it
        loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)

        if self.args.tiler and self.args.output_format == "rtsp":
            # Start streaming
            server = GstRtspServer.RTSPServer.new()
            server.props.service = "%d" % self.args.port
            server.attach(None)

            factory = GstRtspServer.RTSPMediaFactory.new()
            factory.set_launch(
                "( udpsrc name=pay0 port={} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string){}, payload=96 \" )"
                .format(self.args.port, self.args.rtsp_codec))
            factory.set_shared(True)
            server.get_mount_points().add_factory(
                "/{}".format(self.args.output_url), factory)

            self.logger.info("\n *** DeepStream: Launched RTSP Streaming "
                             "at rtsp://localhost:{}/{} ***\n\n".format(
                                 self.args.port, self.args.output_url))
        else:
            server = GstRtspServer.RTSPServer.new()
            server.props.service = "%d" % self.args.port
            server.attach(None)

            factory = []
            for i in range(self.num_sources):
                factory.append(GstRtspServer.RTSPMediaFactory.new())
                factory[i].set_launch(
                    '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
                    % (self.args.port - i - 1, self.args.rtsp_codec))
                factory[i].set_shared(True)
                server.get_mount_points().add_factory(
                    "/{}".format(self.args.output_url) + str(i), factory[i])
                self.logger.info("*** DeepStream: Launched RTSP Streaming "
                                 "at rtsp://localhost:{}/{} ***".format(
                                     self.args.port,
                                     self.args.output_url + str(i)))

        # Start play back and listen to events
        self.logger.info("Starting pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        except:
            pass

        self.logger.info("Exiting pipeline")
        self.pipeline.set_state(Gst.State.NULL)
        self.release()
