def setCWDToProjectDir():
    """
    Set the current working directory to the project root directory.
    """
    import os

    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)

    # Get the directory of the script
    script_dir = os.path.dirname(script_path)

    # Move up to the project root directory
    project_root_dir = os.path.abspath(os.path.join(script_dir, "../.."))

    # Change the current working directory to the project root directory
    os.chdir(project_root_dir)