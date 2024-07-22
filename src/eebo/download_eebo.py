import os
from time import sleep


def download_eebo(path: str = "./dataset/raw_downloads/EEBO.zip"):
    if os.path.exists(path.replace(".zip", "")):
        print(f"EEBO already exists; continue")
        return

    assert " " not in path

    if not os.path.exists("./UMich-eebo_all_compressed.zip") and not os.path.exists(
        path
    ):
        print("EEBO file now found; start downloading")
        sleep(1)
        os.system(
            'lynx -cmd_script=./src/eebo/lynx_eebo_opr.txt "https://drive.google.com/uc?export=download&id=1GMhrPo0TS-CnIc9oRN6sfw3VcUOM1mVO"'
        )
        print("successfully downloaded EEBO")
    else:
        print("EEBO file previously downloaded; reusing")

    if os.path.exists("./UMich-eebo_all_compressed.zip") and not os.path.exists(path):
        os.rename("./UMich-eebo_all_compressed.zip", path)

    print("start unzipping EEBO")
    os.system(
        f'unzip {path} -d {path.replace(".zip", "")} -x "__MACOSX/*" | tqdm --total=60415 > /dev/null'
    )
    middir = "UMich-eebo_all\\ copy/"
    os.system(
        f'mv {os.path.join(path.replace(".zip", ""), middir + "*")} {path.replace(".zip", "")}/'
    )
    os.system(f'rm -r {os.path.join(path.replace(".zip", ""), middir)}')
    print("finished unzipping EEBO")


if __name__ == "__main__":
    download_eebo()
