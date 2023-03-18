from datasets.winequality import files


if not files.have_files():
    print("Missing wine dataset, downloading it")
    files.download_files()