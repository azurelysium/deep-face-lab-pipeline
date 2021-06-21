from functools import partial

import kfp
from kfp import dsl
from kfp.components import func_to_container_op



dfl_op = partial(
    func_to_container_op,
    base_image="ghcr.io/azurelysium/deepfacelab:latest",
    extra_code="""
    import subprocess
    def run_commands(commands, collect=False):
        output = []
        with subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE) as process:
            if process.stdout is not None:
                for line in process.stdout:
                    print(line.strip())
                    if collect:
                        ouptut.append(line.strip())
        return "\n".join(output)
    """
)


## Component: 1. Clear workspace
@dfl_op
def clear_workspace_op() -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 1_clear_workspace.sh
    find /workspace
    """)


## Component: 2.A Download Youtube videos and trim
@partial(
    dfl_op,
    packages_to_intall=["youtube-dl"],
)
def download_videos_op(
        source_youtube_url: str,
        source_start_time: str,
        source_duration: str,
        target_youtube_url: str,
        target_start_time: str,
        target_duration: str,
) -> None:
    run_commands(f"""
    youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' '{source_youtube_url}' -o download.mp4
    ffmpeg -ss {source_start_time} -i download.mp4 -t {source_duration} -c copy /workspace/data_src.mp4

    youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' '{target_youtube_url}' -o download.mp4
    ffmpeg -ss {target_start_time} -i download.mp4 -t {target_duration} -c copy /workspace/data_dst.mp4
    """)


## Component: 2.B Download pre-trained model
@dfl_op
def download_pretrained_model_op() -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 4.1_download_Quick96.sh
    """)

## Component: 2.C Download pre-trained data
@dfl_op
def download_pretrained_data_op() -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 4.1_download_CelebA.sh
    """)


## Component: 3.A Extract images from source video
@dfl_op
def extract_images_from_source_op(
        fps: int = 1,
) -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 2_extract_image_from_data_src.sh --fps {fps} --output-ext png
    """)


## Component: 3.B Extract images from target video
@dfl_op
def extract_images_from_target_op() -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 3_extract_image_from_data_dst.sh --output-ext png
    """)


## Component: 4.A Extract face images from source
@dfl_op
def extract_faces_from_source_op(
        image_size: int = 512,
        jpeg_quality: int = 100,
) -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 4_data_src_extract_faces_S3FD.sh \
         -output-debug \
         --face-type whole_face \
         --max-faces-from-image 1 \
         --image-size {image_size} \
         --jpeg-quality {jpeg_quality} \
    """)


## Component: 4.B Extract face images from target
@dfl_op
def extract_faces_from_target_op(
        image_size: int = 512,
        jpeg_quality: int = 100,
) -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 5_data_dst_extract_faces_S3FD.sh \
         -output-debug \
         --face-type whole_face \
         --max-faces-from-image 0 \
         --image-size {image_size} \
         --jpeg-quality {jpeg_quality} \
    """)


## Component: 5. Train Quick96
@dfl_op
def train_quick96_op(
        timeout: str = "5m",
) -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    timout -s SIGINT {timeout} \
         bash 6_train_Quick96_no_preview.sh \
         --silent-start \
         --force-model-name Quick96
    """)


## Component: 6. Merge Quick96
@dfl_op
def merge_quick96_op(
) -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    echo "0\n" | bash 7_merge_Quick96.sh --force-model-name Quick96
    """)


## Component: 7. Make video output
@dfl_op
def make_video_output_op(
) -> None:
    run_commands(f"""
    cd /app/DeepFaceLab_Linux/scripts/;
    bash 8_merged_to_mp4_lossless.sh
    """)


## Pipeline
@dsl.pipeline(name="DeepFaceLab Pipeline")
def pipeline(
        source_youtube_url: str = "https://www.youtube.com/watch?v=hZNL2j_YJyM",
        source_start_time: str = "00:00:30.00",
        source_duration: str = "00:05:00.00",
        target_youtube_url: str = "https://www.youtube.com/watch?v=-lIPGOD7iSM",
        target_start_time: str = "00:01:15.00",
        target_duration: str = "00:00:05.00",
        workspace_pvc_name: str = "deepfacelab-workspace-pvc",

) -> None:

    # Defining a pipeline

    clear_workspace_task = clear_workspace_op()

    download_pretrained_model_task = download_pretrained_model_op().after(clear_workspace_op)
    download_pretrained_data_task = download_pretrained_data_op().after(clear_workspace_op)
    download_videos_task = download_videos_op(
        source_youtube_url,
        source_start_time,
        source_duration,
        target_youtube_url,
        target_start_time,
        target_duration,
    )
    download_videos_task.after(clear_workspace_task)

    extract_images_from_source_task = extract_images_from_source_op().after(download_videos_task)
    extract_images_from_target_task = extract_images_from_target_op().after(download_videos_task)

    extract_faces_from_source_task = extract_faces_from_source_op().after(extract_images_from_source_task)
    extract_faces_from_target_task = extract_faces_from_target_op().after(extract_images_from_target_task)

    train_quick96_task = train_quick96_op().after(
        download_pretrained_data_op,
        download_pretrained_model_op,
        extract_faces_from_source_task,
    )
    merge_quick96_task = merge_quick96_op().after(train_quick96_task)
    make_video_output_task = make_video_output_op().after(merge_quick96_task)


    # Attach volumes and disable caching
    vop = dsl.VolumeOp(
        name="volume_creation",
        resource_name=workspace_pvc_name,
        size="16Gi"
    )

    def op_transformer(op):
        op.add_pvolumes({"/workspace": vop.volume}))
        op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    pipeline_conf = kfp.dsl.get_pipeline_conf()
    pipeline_conf.add_op_transformer(op_transformer)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile("pipeline.yaml")
