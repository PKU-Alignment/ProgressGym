"""
Project Gutenberg parsing with python 3.

Written by
M. Gerlach & F. Font-Clos

"""

from src.gutenberg.src.utils import populate_raw_from_mirror, list_duplicates_in_mirror
from src.gutenberg.src.metadataparser import make_df_metadata
from src.gutenberg.src.bookshelves import get_bookshelves
from src.gutenberg.src.bookshelves import parse_bookshelves

import argparse
import os
import subprocess
import pickle


def get_data_gutenberg(dir: str):

    class args:
        mirror = os.path.join(dir, "data/.mirror/")
        raw = os.path.join(dir, "data/raw/")
        metadata = os.path.join(dir, "metadata/")
        pattern = "*"
        keep_rdf = False
        overwrite_raw = False
        quiet = False

    if os.path.isdir(args.raw):
        print("Gutenberg already downloaded and cleaned. Skipping to next stage.")
        return

    # create directories if they don't exist
    mirror_exists = True
    if not os.path.isdir(args.mirror):
        os.makedirs(args.mirror)
        mirror_exists = False

    if not os.path.isdir(args.raw):
        os.makedirs(args.raw)
    if not os.path.isdir(args.metadata):
        os.makedirs(args.metadata)

    # Update the .mirror directory via rsync
    # --------------------------------------
    # We sync the 'mirror_dir' with PG's site via rsync
    # The matching pattern, explained below, should match
    # only UTF-8 files.

    # pass the -v flag to rsync if not in quiet mode
    if args.quiet:
        vstring = ""
    else:
        vstring = "v"

    # Pattern to match the +  but not the - :
    #
    # + 12345 .   t   x  t .            utf  8
    # - 12345 .   t   x  t .      utf8 .gzi  p
    # + 12345 -   0   .  t x                 t
    # ---------------------------------------------
    #        [.-][t0][x.]t[x.]    *         [t8]
    sp_args = [
        "rsync",
        "-am%s" % vstring,
        "--include",
        "*/",
        "--include",
        "[p123456789][g0123456789]%s[.-][t0][x.]t[x.]*[t8]" % args.pattern,
        "--exclude",
        "*",
        "aleph.gutenberg.org::gutenberg",
        args.mirror,
    ]
    if not mirror_exists:
        subprocess.call(sp_args)
    else:
        print(
            f"Gutenberg: Gutenberg files already downloaded from mirror (are they complete?); skipping the downloading stage"
        )

    # Get rid of duplicates
    # ---------------------
    # A very small portion of books are stored more than
    # once in PG's site. We keep the newest one, see
    # erase_duplicates_in_mirror docstring.
    dups_list = list_duplicates_in_mirror(mirror_dir=args.mirror)

    # Populate raw from mirror
    # ------------------------
    # We populate 'raw_dir' hardlinking to
    # the hidden 'mirror_dir'. Names are standarized
    # into PG12345_raw.txt form.
    populate_raw_from_mirror(
        mirror_dir=args.mirror,
        raw_dir=args.raw,
        overwrite=args.overwrite_raw,
        dups_list=dups_list,
        quiet=args.quiet,
    )

    # Update metadata
    # ---------------
    # By default, update the whole metadata csv
    # file each time new data is downloaded.
    make_df_metadata(
        path_xml=os.path.join(args.metadata, "rdf-files.tar.bz2"),
        path_out=os.path.join(args.metadata, "metadata.csv"),
        update=args.keep_rdf,
    )

    # Bookshelves
    # -----------
    # Get bookshelves and their respective books and titles as dicts
    BS_dict, BS_num_to_category_str_dict = parse_bookshelves(args.metadata)
    with open(os.path.join(args.metadata, "bookshelves_ebooks_dict.pkl"), "wb") as fp:
        pickle.dump(BS_dict, fp)
    with open(
        os.path.join(args.metadata, "bookshelves_categories_dict.pkl"), "wb"
    ) as fp:
        pickle.dump(BS_num_to_category_str_dict, fp)
