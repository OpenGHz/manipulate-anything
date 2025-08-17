import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from rlbench.utils import get_rgb_mask


def extract_rgb_mask(
    data_root,
    task_name,
    peak_choices: list,
    show=False,
    save=True,
    unique_gray=False,
    rgb_choices=None,
    camera_predix=("front",),
):
    ep_dir = f"{data_root}/{task_name}/all_variations/episodes"
    ep_length = len(list(Path(ep_dir).glob("episode*")))
    for ep_index in range(0, ep_length):
        episode = Path(ep_dir) / f"episode{ep_index}"
        print(f"Processing {episode}...")
        for prefix in camera_predix:
            rgb_length = len(list(episode.glob(f"{prefix}_rgb/*.png")))
            mask_length = len(list(episode.glob(f"{prefix}_mask/*.png")))
            assert rgb_length == mask_length, (
                f"RGB ({rgb_length}) and mask ({mask_length}) counts do not match in {episode}"
            )

        for prefix in camera_predix:
            if save:
                output_dir = episode / f"{prefix}_rgb_mask"
                output_dir.mkdir(parents=True, exist_ok=True)
            for index in range(0, rgb_length):
                rgb = episode / f"{prefix}_rgb" / f"{index}.png"
                mask = episode / f"{prefix}_mask" / f"{index}.png"
                raw_mask_image = cv2.imread(str(mask))
                raw_rgb_image = cv2.imread(str(rgb))

                masked_image, gray_image, hist, peaks, peak_choices = get_rgb_mask(
                    raw_rgb_image, raw_mask_image, peak_choices, rgb_choices
                )
                print(f"{index}: {len(peaks)} Peaks found at:", peaks)
                print(f"Using peak choices: {peak_choices}")
                if show:
                    plt.plot(hist)
                    cv2.imshow("RGB Image", raw_rgb_image)
                    cv2.imshow("Gray Image", gray_image)
                    cv2.imshow("Original Image", raw_mask_image)
                    cv2.imshow("Masked RGB Image", masked_image)
                    plt.xlim([0, 256])
                    plt.show(block=False)
                    if cv2.waitKey(0) == 27:  # Wait for ESC key to exit
                        cv2.destroyAllWindows()
                        exit(0)
                if save:
                    output_file = (output_dir / f"{index}.png").absolute()
                    # print(output_file)
                    cv2.imwrite(str(output_file), masked_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="/home/ghz/Work/Research/data/GT_12T_data/train_12_GT",
    )
    args = parser.parse_args()

    data_root = args.data_root

    task_peaks = {
        "lamp_on": [(7, 14), slice(4, None)],
        # "open_box": [(7, 14), slice(4, None)],
        # "open_wine_bottle": [(7, 14), slice(4, None)],
        # "pick_and_lift": [(7, 14)],
        # "pick_up_cup": [(7, 14)],
        # "play_jenga": [(7, 14), slice(4, None)],
        # "put_knife_on_chopping_board": [(7, 14), slice(4, None)],
        # "slide_block_to_target": [(7, 14), slice(4, None)],
        # "take_umbrella_out_of_umbrella_stand": [(7, 14), slice(4, None)],
        # ignore the tasks below
        # "put_objects_in_container": [(7, 14), slice(4, None)],  # task demos are not so good
        # "open_jar": [(7, 14), slice(4, None)],  # mask is hard to separate
    }

    # unique_gray = {"pick_and_lift": True}
    rgb_choices = {
        "pick_and_lift": [((0, 0, 135), (0, 0, 255))],
        "pick_up_cup": [((0, 0, 110), (0, 0, 255))],
    }
    # rgb_choices = {}

    # task_name = "close_box"
    # extract_rgb_mask(data_root, task_name, [(11, 14), (37, 41)], show=False, save=True)

    for task_name in task_peaks:
        extract_rgb_mask(
            data_root,
            task_name,
            task_peaks[task_name],
            show=False,
            save=True,
            rgb_choices=rgb_choices.get(task_name, None),
        )
