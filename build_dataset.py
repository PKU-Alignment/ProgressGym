import src.text_writer as tw
import src.cleanser.rule_based_cleanser as rb
import src.cleanser.localllm_cleanser as llm_cleanser
import src.model_training.train_hislm as hislm
import pdb
import traceback
import sys
import os
import time


import src.eebo.download_eebo as eebo_dl
import src.eebo.process_eebo as eebo_pc


def build_EEBO():
    print("======= START BUILDING EEBO DATASET =======")
    eebo_dl.download_eebo()
    eebo_pc.build_eebo_dataset()
    print("======= FINISHED BUILDING EEBO DATASET =======\n\n\n")


import src.gutenberg.get_data as gtb_gd
import src.gutenberg.get_meta as gtb_gm


def build_gutenberg():
    print("======= START BUILDING GUTENBERG DATASET =======")
    dir = "./dataset/raw_downloads/Gutenberg/"
    gtb_gd.get_data_gutenberg(dir)
    gtb_gm.gather_meta(
        os.path.join(dir, "data/raw"), "./dataset/raw_downloads/Gutenberg_records.txt"
    )
    print("======= FINISHED BUILDING GUTENBERG DATASET =======\n\n\n")


import src.internet_archive.get_sources as ia_gs


def build_internet_archive(max_hours: int = None):
    print("======= START BUILDING INTERNET ARCHIVE DATASET =======")
    ia_gs.build_internet_archive_LibOfCong(max_hours)
    print("======= FINISHED BUILDING INTERNET ARCHIVE DATASET =======\n\n\n")


import src.pile_of_law.get_data as pol_gd


def build_pile_of_law():
    print("======= START BUILDING PILE OF LAW DATASET =======")
    pol_gd.get_pile_of_law()
    print("======= FINISHED BUILDING PILE OF LAW DATASET =======\n\n\n")


if __name__ == "__main__":
    tw.write_log(f"\n\n\n\n\n\n=========== NEW RUN ============\n\n")

    try:
        proceed = False
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        if (
            "y"
            in input(
                "Build dataset from scratch? If yes, execute `sudo apt install lynx` (used for downloading data files) before proceeding. (y/n)"
            ).lower()
        ):
            proceed = True

            build_gutenberg()  # takes ~5h, download ~35GB
            build_EEBO()  # takes ~1.5h, download ~3.5GB
            build_pile_of_law()  # takes ~5h
            build_internet_archive(
                max_hours=10
            )  # takes ~100h, but if max_hours is supplied then stops after this many hours (won't affect data integrity)
            # finishing up
            tw.seal_all_files()
            print("Finished building entire dataset. Proceed to data cleansing.")

        if (
            proceed
            or "y" in input("Perform rule-based dataset cleansing? (y/n)").lower()
        ):
            proceed = True
            rb.cleanse(
                "./dataset/dataset_text_sequence/",
                "./dataset/dataset_text_sequence_rulebased_cleansed/",
            )
            print("Finished rule-based data cleansing. Now exiting.")

        if (
            proceed
            or "y" in input("Perform LLM-based dataset cleansing? (y/n)").lower()
        ):
            proceed = True
            llm_cleanser.run_cleanser(
                in_path="./dataset/dataset_text_sequence_rulebased_cleansed/",
                out_path="./dataset/dataset_text_sequence_llm_cleansed/",
            )

            # Make llm-cleansed version the official version ("dataset_text_sequence"), and move the other two versions into dataset/raw_downloads
            path = (
                f"./dataset/raw_downloads/dataset_text_sequence_versions/{timestamp}/"
            )
            os.makedirs(path)

            print(f"Moving pre-cleansing version to backup folder...")
            os.rename(
                "./dataset/dataset_text_sequence/",
                os.path.join(path, "dataset_text_sequence_original/"),
            )

            print(f"Moving rule-cleansed version to backup folder...")
            os.rename(
                "./dataset/dataset_text_sequence_rulebased_cleansed/",
                os.path.join(path, "dataset_text_sequence_rulebased_cleansed/"),
            )

            print(f"Copying LLM-cleansed version to official position...")
            os.system(
                f"cp -r ./dataset/dataset_text_sequence_llm_cleansed/ ./dataset/dataset_text_sequence/"
            )

            print(f"Copying LLM-cleansed version to backup folder...")
            os.rename(
                "./dataset/dataset_text_sequence_llm_cleansed/",
                os.path.join(path, "dataset_text_sequence_llm_cleansed/"),
            )

            print("Finished LLM-based data cleansing. Exiting.")

        if proceed or "y" in input("Curate prompts for model training? (y/n)").lower():
            proceed = True
            import src.model_training.curate_prompts as cp

            cp.curate_prompts()
            print("Finished prompt curation. Exiting.")

        if proceed or "y" in input("Perform model training? (y/n)").lower():
            proceed = True

            print(f"Removing overly small or messy subdatasets...")
            path = f"./dataset/raw_downloads/dataset_text_sequence_versions/{timestamp}/removed/"
            os.makedirs(path)

            sub_datasets = [
                f
                for f in os.listdir("./dataset/dataset_text_sequence/")
                if os.path.isdir(os.path.join("./dataset/dataset_text_sequence/", f))
            ]
            for sub in sub_datasets:
                # Remove if size < 10MB AND century number < 13
                if (
                    hislm.get_directory_size_bytes(
                        os.path.join("./dataset/dataset_text_sequence/", sub)
                    )
                    < 10 * 1024 * 1024
                    and int(sub.strip("C")) < 13
                ):
                    print(f"Removing {sub}...")
                    os.system(f"mv ./dataset/dataset_text_sequence/{sub} {path}")

            hislm.run_training(
                "./dataset/dataset_text_sequence/", "./dataset/dataset_model_sequence/"
            )
            print("Finished model training. Exiting.")

    except:
        print(f"Exception occured. Entering debugger.")

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
