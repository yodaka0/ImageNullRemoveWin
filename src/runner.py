import os
import shutil
import time
from typing import Union

from omegaconf import OmegaConf

#from src.run_clip import clip
from src.run_cls import classifire_predict
from src.run_megadetector import run_mdet_crop, run_megadetector
from src.run_summary import img_cls_summary, video_cls_summary
from src.utils.config import (
    ClipConfig,
    ClsConfig,
    MDetConfig,
    MDetCropConfig,
    MDetRenderConfig,
    RootConfig,
    SummaryConfig,
)
from src.utils.logger import get_logger
from src.utils.tag import SessionTag, session_tag_list
from src.utils.timer import Timer


class Runner:
    def __init__(
        self,
        rconfig: RootConfig,
        mconfig: MDetConfig,
        sconfig: SummaryConfig,
        session_tag: Union[SessionTag, list[SessionTag], str, list[str]],
        folders = None,
    ) -> None:
        self.__check_config(rconfig)
        if isinstance(session_tag, (str, list)):
            self.session_tags: list[SessionTag] = (
                [SessionTag.value_of(tag) for tag in session_tag]
                if isinstance(session_tag, list)
                else [SessionTag.value_of(session_tag)]
            )
        else:
            self.session_tags: list[SessionTag] = (
                session_tag if isinstance(session_tag, list) else [session_tag]
            )

        _session_tags_str = "-".join(
            [session_tag.name for session_tag in self.session_tags]
        )
        self.logdir = rconfig.log_dir.joinpath(
            f'{time.strftime("%Y%m%d%H%M%S")}_{_session_tags_str}_{rconfig.session_root.name}'
        )
        os.makedirs(self.logdir, exist_ok=True)
        self.logger = get_logger(out_dir=self.logdir, logname=f"{session_tag}.log")
        OmegaConf.save(config=rconfig, f=self.logdir.joinpath("config.raw.yaml"))
        self.rconfig = rconfig
        self.mconfig = mconfig
        self.sconfig = sconfig
        self.folders = folders

    def exec_mdet(self, mconfig: MDetConfig, folders) -> None:
        input_file_path = mconfig.image_source if mconfig.image_source.is_file() else None
        output_file_path = (
            mconfig.image_source
            if mconfig.image_source.is_dir()
            else mconfig.image_source.parent
        ).joinpath("detector_output.json")
        self.logger.info(f"Start {mconfig.image_source} MegaDetector Detection...")
        self.logger.info(f"Output file: {output_file_path}")
        output=run_megadetector(detector_config=mconfig,folders=folders)
        self.logger.info("Detection Complete")
        shutil.copyfile(
            str(output_file_path), str(self.logdir.joinpath(output_file_path.name))
        )
        if input_file_path is not None:
            shutil.copyfile(
                str(input_file_path), str(self.logdir.joinpath(input_file_path.name))
            )
        return output

    """def exec_mdet_crop(self, config: MDetCropConfig) -> None:
        self.logger.info(f"Start {config.image_source} MegaDetector Cropping...")
        self.logger.info(f"MDet Output file: {config.mdet_result_path}")
        self.logger.info(f"Output file: {config.output_dir}")
        if config.mdet_result_path is None:
            raise ValueError(
                f"Invalid Value of mdet_result_path: {config.mdet_result_path}. Please enter."
            )
        if config.output_dir is None:
            raise ValueError(
                f"Invalid Value of output_dir: {config.output_dir}. Please enter."
            )
        run_mdet_crop(config=config)
        self.logger.info("MDet cropping Complete!")"""

    def exec_mdet_render(self, config: MDetRenderConfig) -> None:
        pass

    """def exec_clip(self, config: ClipConfig) -> None:
        self.logger.info(f"Start {config.video_dir.name} Clopping...")
        self.logger.info(f"Save Dir: {config.output_dir}")
        _end_frame = config.end_frame if config.end_frame is not None else "end"
        self.logger.info(f"Frame: {config.start_frame}-{_end_frame}")
        self.logger.info(f"Remove Banner: {config.remove_banner}")
        with Timer(verbose=True, logger=self.logger, timer_tag="Clip"):
            clip(config)
        self.logger.info("Clip Complete!")"""

    def exec_cls(self, config: ClsConfig) -> None:
        input_file_path = config.image_source if config.image_source.is_file() else None
        self.logger.info(f"Start {config.image_source} Classifire Prediction...")
        classifire_predict(cls_config=config)
        self.logger.info(f"Result file: {config.result_file_name}")
        if input_file_path is not None:
            shutil.copyfile(
                str(input_file_path.parent.joinpath(config.result_file_name)),
                str(self.logdir.joinpath(config.result_file_name)),
            )
            shutil.copyfile(
                str(input_file_path), str(self.logdir.joinpath(input_file_path.name))
            )
        else:
            shutil.copyfile(
                str(config.image_source.joinpath(config.result_file_name)),
                str(self.logdir.joinpath(config.result_file_name)),
            )
        self.logger.info("Prediction Complete!")

    def exec_img_summary(self, sconfig: SummaryConfig) -> None:
        self.logger.info(f"Start {sconfig.cls_result_dir} Classifire Summarize...")
        # self.logger.info(f"Summarized file: {sconfig.img_summary_name}")
        result_path = img_cls_summary(config=sconfig)
        self.logger.info(f"Result file: {result_path}")
        shutil.copyfile(
            str(result_path),
            str(self.logdir.joinpath(result_path.name)),
        )
        shutil.copyfile(
            str(sconfig.cls_result_dir.joinpath(sconfig.img_summary_name)),
            str(self.logdir.joinpath(sconfig.img_summary_name)),
        )
        self.logger.info("IMG wise Summary Complete!")
        if sconfig.is_video_summary:
            sequence_result_path = video_cls_summary(config=sconfig)
            self.logger.info(f"Sequence Result file: {sequence_result_path}")
            shutil.copyfile(
                str(sequence_result_path),
                str(self.logdir.joinpath(sequence_result_path.name)),
            )
            self.logger.info("MOVIE wise Summary Complete!")

    def __check_config(self, config: RootConfig) -> None:
        assert (
            config.session_root.exists()
        ), f"{config.session_root} does not exist. Please enter the path where it exists."
        assert os.access(
            str(config.session_root), os.R_OK
        ), f"{config.session_root} is not readable. Please enter the path where it is readable."
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            assert os.access(
                str(config.output_dir), os.W_OK
            ), f"{config.output_dir} is not writable. Please enter the path where it is writable."
        assert (
            config.session_root.is_absolute()
        ), f"{config.session_root} is not an absolute path. Please enter the absolute path."

    def __drop_config(self, rconfig: RootConfig, exec_list: dict) -> RootConfig:
        if exec_list[SessionTag.MDet] is False:
            rconfig = None
        if exec_list[SessionTag.MDetCrop] is False:
            rconfig.mdet_crop_config = None
        if exec_list[SessionTag.MDetRender] is False:
            rconfig.mdet_render_config = None
        if exec_list[SessionTag.Clip] is False:
            rconfig.clip_config = None
        if exec_list[SessionTag.Cls] is False:
            rconfig.cls_config = None
        return rconfig

    def __exec_session(
        self,
        rconfig: RootConfig,
        mconfig: MDetConfig,
        sconfig: SummaryConfig,
        session_tag: SessionTag,
        folders = None,
    ) -> None:
        self.logger.info(session_tag)
        if session_tag == SessionTag.MDet:
            if mconfig is not None:
                self.exec_mdet(mconfig=mconfig,folders=folders)
        elif session_tag == SessionTag.MDetCrop:
            if rconfig.mdet_crop_config is not None:
                self.exec_mdet_crop(config=rconfig.mdet_crop_config)
        elif session_tag == SessionTag.MDetRender:
            if rconfig.mdet_render_config is not None:
                self.exec_mdet_render(config=rconfig.mdet_render_config)
        elif session_tag == SessionTag.Clip:
            if rconfig.clip_config is not None:
                self.exec_clip(config=rconfig.clip_config)
        elif session_tag == SessionTag.Cls:
            if rconfig.cls_config is not None:
                self.exec_cls(config=rconfig.cls_config)
        elif session_tag == SessionTag.ImgSummary:
            if sconfig is not None:
                self.exec_img_summary(sconfig=sconfig)

    def execute(self) -> None:
        exec_list = {k: False for k in session_tag_list}

        with Timer(verbose=True, logger=self.logger, timer_tag="AllProcess"):
            for session_tag in self.session_tags:
                self.__exec_session(rconfig=self.rconfig, mconfig=self.mconfig, sconfig=self.sconfig, session_tag=session_tag,folders=self.folders)
                exec_list[session_tag] = True

        #exec_config = self.__drop_config(rconfig=self.rconfig,mconfig=self.mconfig,sconfig=self.sconfig, exec_list=exec_list)
        #OmegaConf.save(config=exec_config, f=self.logdir.joinpath("config.yaml"))
