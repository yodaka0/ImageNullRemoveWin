from omegaconf import OmegaConf

if __name__ == '__main__':
    from src.runner import Runner
from src.utils.config import RootConfig, MDetConfig, SummaryConfig
from multiprocessing import Process
import multiprocessing
from threading import Thread
import sys
import time
import src.megadetector.visualization.visualization_utils as viz_utils
from src.megadetector.detection.run_detector import FAILURE_IMAGE_OPEN, FAILURE_INFER, load_detector
import humanfriendly
import os
import argparse
from PIL import Image, ImageDraw
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("session_root", type=str, help="The root directory for the session")

    args = parser.parse_args()
    session_root = args.session_root

    # ここで session_root 変数を使用できます
    print(f"Session root is: {session_root}")

    return session_root

def create_new_structure(src_dir, dst_dir):
    for dir, _ ,_ in os.walk(src_dir):
        dirs_name = dir.replace(dst_dir, "")
        new_dir = dst_dir + "\\" + dirs_name.replace("\\", "_out\\") + "_out"
        os.makedirs(new_dir, exist_ok=True)

# from src.utils.tag import SessionTag
if __name__ == '__main__':
    session_root=main()
    mconfig = OmegaConf.structured(
        MDetConfig(image_source=session_root)
    )
    sconfig = OmegaConf.structured(
        SummaryConfig()
    )
    rconfig = OmegaConf.structured(
        RootConfig(
            session_root=session_root, output_dir=session_root,
        )
    )
    parent_dir = os.path.dirname(session_root)+"\\"
    create_new_structure(session_root, parent_dir)
    runner = Runner(mconfig=mconfig, sconfig=sconfig, rconfig=rconfig, session_tag="mdet",folders=session_root)
    runner.execute()

# Number of images to pre-fetch
max_queue_size = 10
use_threads_for_queue = True
verbose = False

def process_image(im_file, detector, confidence_threshold, image=None,
                  quiet=False, image_size=None,folders=None):
    """
    Runs MegaDetector over a single image file.

    Args
    - im_file: str, path to image file
    - detector: loaded model
    - confidence_threshold: float, only detections above this threshold are returned
    - image: previously-loaded image, if available

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """

    if not quiet:
        print('Processing image {}'.format(im_file))

    if image is None:
        try:
            image = viz_utils.load_image(im_file)
        except Exception as e:
            if not quiet:
                print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': FAILURE_IMAGE_OPEN
            }
            return result
    try:
        result, object, bbox = detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold, image_size=image_size,folders=folders) 
        try:
            if object > 0 :  
                folder = os.path.dirname(folders)+"\\"
                new_folder = im_file.replace(folder,"").replace(".JPG","_bb.JPG")
                ex_file =os.path.basename(new_folder)
                new_file = folder + new_folder.replace("\\","_out\\")
                draw = ImageDraw.Draw(image)
                for b in bbox:
                    image_width, image_height = image.size
                    image_bbox = [
                            b[0] * image_width,  # x0
                            b[1] * image_height, # y0
                            (b[0]+b[2]) * image_width,  # x1
                            (b[1]+b[3]) * image_height  # y1
                            ]
                    draw.rectangle(image_bbox, outline='red')
                
                if os.path.exists(new_file):
                    print(f"{new_file} is exists")
                else:
                    print(new_file)
                    image.save(new_file)
        except Exception as e:
            object = 0
        exif_data = image._getexif()
        #result["ModifyDate"] = exif_data[306]
        date, time = exif_data[36867].split(' ')
        result["Date"] = date
        result["Time"] = time
        result["Make"] = exif_data[271]
        #result["Model"] = exif_data[272]
        result["object"] = object
        try:
            if object > 0 :  
                result["extract_file"] = ex_file
        except Exception as e:
            result["extract_file"] = ""
    except Exception as e:
        if not quiet:
            print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': FAILURE_INFER,
            'object': 0
        }
        return result

    return result

#%% Support functions for multiprocessing

def producer_func(q,image_files):
    """
    Producer function; only used when using the (optional) image queue.

    Reads up to N images from disk and puts them on the blocking queue for processing.
    """

    if verbose:
        print('Producer starting'); sys.stdout.flush()

    for im_file in image_files:

        try:
            if verbose:
                print('Loading image {}'.format(im_file)); sys.stdout.flush()
            image = viz_utils.load_image(im_file)
        except Exception as e:
            print('Producer process: image {} cannot be loaded. Exception: {}'.format(im_file, e))
            raise

        if verbose:
            print('Queueing image {}'.format(im_file)); sys.stdout.flush()
        q.put([im_file,image])

    q.put(None)

    print('Finished image loading'); sys.stdout.flush()


def consumer_func(q,return_queue,model_file,confidence_threshold,image_size=None,folders=None):
    """
    Consumer function; only used when using the (optional) image queue.

    Pulls images from a blocking queue and processes them.
    """

    if verbose:
        print('Consumer starting'); sys.stdout.flush()

    start_time = time.time()
    print(start_time)
    detector = load_detector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model (before queueing) in {}'.format(humanfriendly.format_timespan(elapsed)))
    sys.stdout.flush()

    results = []

    n_images_processed = 0

    while True:
        r = q.get()
        if r is None:
            q.task_done()
            return_queue.put(results)
            return
        n_images_processed += 1
        im_file = r[0]
        image = r[1]
        if verbose or ((n_images_processed % 10) == 0):
            elapsed = time.time() - start_time
            images_per_second = n_images_processed / elapsed
            print('De-queued image {} ({}/s) ({})'.format(n_images_processed,
                                                          images_per_second,
                                                          im_file));
            sys.stdout.flush()
        results.append(process_image(im_file=im_file,detector=detector,
                                     confidence_threshold=confidence_threshold,
                                     image=image,quiet=True,image_size=image_size,
                                     folders=folders))
        if verbose:
            print('Processed image {}'.format(im_file)); sys.stdout.flush()
        q.task_done()

def run_detector_with_image_queue(image_files,model_file,confidence_threshold,
                                  quiet=False,image_size=None,folders=None):
    """
    Driver function for the (optional) multiprocessing-based image queue; only used when --use_image_queue
    is specified.  Starts a reader process to read images from disk, but processes images in the
    process from which this function is called (i.e., does not currently spawn a separate consumer
    process).
    """
    try:
        q = multiprocessing.JoinableQueue(maxsize=max_queue_size)
        return_queue = multiprocessing.Queue(1)

        if use_threads_for_queue:
            producer=Thread(target=producer_func,args=(q,image_files))
            print('Using threads for queue')
        else:
            producer=Process(target=producer_func,args=(q,image_files))
            print('Using processes for queue')
        producer.daemon = False
        producer.start()

        # TODO
        #
        # The queue system is a little more elegant if we start one thread for reading and one
        # for processing, and this works fine on Windows, but because we import TF at module load,
        # CUDA will only work in the main process, so currently the consumer function runs here.
        #
        # To enable proper multi-GPU support, we may need to move the TF import to a separate module
        # that isn't loaded until very close to where inference actually happens.
        run_separate_consumer_process = False

        if run_separate_consumer_process:
            if use_threads_for_queue:
                consumer = Thread(target=consumer_func,args=(q,return_queue,model_file,
                                                            confidence_threshold,image_size,
                                                            folders,))
            else:
                consumer = Process(target=consumer_func,args=(q,return_queue,model_file,
                                                            confidence_threshold,image_size,
                                                            folders))
            consumer.daemon = True
            consumer.start()
        else:
            consumer_func(q,return_queue,model_file,confidence_threshold,image_size,folders)

        producer.join()
        print('Producer finished')

        if run_separate_consumer_process:
            consumer.join()
            print('Consumer finished')
        else:
            print('Consumer ended')

        q.join()
        print('Queue joined')

        if not return_queue.empty():
            results = return_queue.get()

            return results
        
        else:
            print('Warning: no results returned from queue')
            return []
    except Exception as e:
        print('Exception: {}'.format(e))
        raise
