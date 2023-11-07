import json
import os
from concurrent import futures
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from src.megadetector.data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name,
)
from src.megadetector.detection.run_detector_batch import (
    load_and_run_detector_batch,
    write_results_to_file,
)
from src.megadetector.visualization import visualization_utils as vis_utils
from src.utils import image_pathlist_load_from_file
from src.utils.config import MDetConfig, MDetCropConfig
from src.utils.timer import Timer

DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

logger = getLogger(__file__)


def run_megadetector(
    detector_config: MDetConfig,
    output_json_name: str = "detector_output.json",
    folders = None,
) -> None:
    image_source: Path = detector_config.image_source
    image_data_dir = image_source if image_source.is_dir() else image_source.parent

    assert (
        detector_config.model_path.exists()
    ), f"detector file {str(detector_config.model_path)} does not exist"
    assert (
        0.0 < detector_config.threshold <= 1.0
    ), "Confidence threshold needs to be between 0 and 1"  # Python chained comparison
    assert output_json_name.endswith(
        ".json"
    ), "output_file specified needs to end with .json"
    if not detector_config.output_absolute_path:
        assert (
            image_source.is_dir()
        ), "image_file must be a directory when megadetector.output_absolute_path is not True"

    # if image_source.is_dir():
    #     image_file_names = ImagePathUtils.find_images(str(image_source),
    #                                                   detector_config.recursive)
    #     image_file_names = [
    #         image_file_name
    #         for image_file_name in image_file_names
    #         if not os.path.join(image_source, "exept_dif") in image_file_name
    #     ]
    #     logger.info("{} image files found in the input directory".format(len(image_file_names)))
    # # A json list of image paths
    # elif image_source.is_file() and image_source.suffix == ".json":
    #     with open(image_source) as f:
    #         image_file_names = json.load(f)
    #     logger.info("{} image files found in the json list".format(len(image_file_names)))
    # elif image_source.is_file() and image_source.suffix == ".csv":
    #     df = pd.read_csv(str(image_source), header=0)
    #     image_file_names = df["fullpath"].to_list()
    #     logger.info("{} image files found in the csv list".format(len(image_file_names)))
    # # A single image file
    # elif image_source.is_file and ImagePathUtils.is_image_file(str(image_source)):
    #     image_file_names = [image_source]
    #     logger.info("A single image at {} is the input file".format(image_source))
    #     # photo_data_dir = image_source.parent
    # else:
    #     raise ValueError(
    #         "image_source specified is not a directory, a json list, or an image file, "
    #         "(or does not have recognizable extensions)."
    #     )
    image_file_names = image_pathlist_load_from_file(
        image_source, detector_config.recursive
    )

    assert (
        len(image_file_names) > 0
    ), "Specified image_source does not point to valid image files"
    assert os.path.exists(
        image_file_names[0]
    ), f"The first image to be scored does not exist at {image_file_names[0]}"

    logger.info(f"Photo data directory contains {len(image_file_names)} images.")

    results = []
    with Timer(timer_tag="MegaDetector", verbose=True, logger=logger):
        results = load_and_run_detector_batch(
            model_file=str(detector_config.model_path),
            image_file_names=image_file_names,
            checkpoint_path=None,
            confidence_threshold=detector_config.threshold,
            checkpoint_frequency=-1,
            results=results,
            n_cores=detector_config.ncores,
            use_image_queue=True,
            quiet=not detector_config.verbose,
            folders=folders,
        )

    logger.info(f"Finished inference for {len(results)} images.")

    relative_path_base = None
    if not detector_config.output_absolute_path:
        relative_path_base = str(image_data_dir)
    output=write_results_to_file(
        results,
        str(image_data_dir.joinpath(output_json_name)),
        relative_path_base=relative_path_base,
        detector_file=str(detector_config.model_path),
        folders=folders,
    )
    return output


def run_mdet_crop(config: MDetCropConfig) -> list[Path]:
    assert config.mdet_result_path is not None, (
        "Please enter the mdet_result_path, "
        "ex) python run_mdet_crop.py session_root=*** "
        "output_dir=*** mdet_result_path=***"
    )
    assert (
        config.threshold > 0 and config.threshold < 1
    ), f"Confidence threshold {config.threshold} is invalid, must be in (0, 1)."
    assert (
        config.mdet_result_path.exists()
    ), f"MegaDetector output file does not exist at {config.mdet_result_path}."

    with open(config.mdet_result_path) as f:
        detector_output = json.load(f)
    assert (
        "images" in detector_output
    ), 'Detector output file should be a json with an "images" field.'
    images = detector_output["images"]
    num_images = len(images)
    logger.info(f"Detector output file contains {num_images} entries.")
    logger.info(
        "Cropping detections above a confidence threshold of {}...".format(
            config.threshold
        )
    )
    num_saved = 0
    croped_img_paths = []

    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tasks = [
            executor.submit(
                crop,
                entry=entry,
                output_dir=config.output_dir,
                image_dir=config.image_source,
                threshold=config.threshold,
            )
            for entry in images
        ]
        for future in tqdm(futures.as_completed(tasks), total=len(images)):
            crop_flag, croped_img_path = future.result()
            num_saved += crop_flag
            if croped_img_path is None:
                pass
            else:
                croped_img_paths.append(croped_img_path)
    logger.info(
        f"Cropping detection results on {num_saved} images, "
        f"saved to {config.output_dir}."
    )
    # src_filepaths = [
    #     config.mdet_result_path.parent.joinpath(entry["file"]).absolute()
    #     for entry in images
    # ]
    # pd.DataFrame(
    #     [src_filepaths, [None] * len(src_filepaths), [None] * len(src_filepaths)],
    #     index=["filepath", "substance", "n_bbox"],
    # ).T.to_csv(config.output_dir.joinpath("img_wise_cls_summary.csv"), index=None)
    return croped_img_paths


def crop(
    entry: dict[Any], output_dir: Path, image_dir: Path, threshold: float
) -> tuple[int, Optional[Path]]:
    image_id = entry["file"]
    if entry["max_detection_conf"] < threshold:
        return 0, None
    if "failure" in entry:
        logger.info(f'Skipping {image_id}, failure: "{entry["failure"]}"')
        return 0, None
    if Path(image_id).is_absolute():
        image_obj = Path(image_id)
    else:
        image_obj = image_dir.joinpath(image_id)

    if not image_obj.exists():
        logger.info(f"Image {image_id} not found in images_dir; skipped.")
        return 0, None

    image = vis_utils.open_image(image_obj)
    images_cropped = vis_utils.crop_image(
        entry["detections"], image, confidence_threshold=threshold
    )
    image_parts = [
        parts for parts in list(image_obj.parts) if parts not in image_dir.parts
    ]
    save_dir = output_dir.joinpath("/".join(image_parts[:-1]))
    if not save_dir.exists():
        os.makedirs(save_dir, exist_ok=True)
    for i_crop, cropped_image in enumerate(images_cropped):
        img_name, ext = Path(image_id).stem, Path(image_id).suffix
        crop_img_path = save_dir.joinpath(f"{img_name}---{i_crop}{ext}")
        cropped_image.save(crop_img_path)
    return 1, crop_img_path
