"""A submodule for generating the _data_ids.yaml database of downloads.

This submodule is used for maintenance of the package and the data availeble
for download from Box, while it is included as part of the package, it is not
intended to be used by the user. It is used to generate the _data_ids.yaml
file, which will be updated in new releases as and when new data is added to
the database.

This requires the Box SDK to be installed. You can install it with
`pip install boxsdk`. You will also need to set the environment variables
`SYNTH_BOX_ID` and `SYNTH_BOX_SECRET` to the client ID and secret of your
Box application. You can create a Box application at
https://developer.box.com/console/apps (assuming you have a Box account with
edit permissions to the folder).

Once you run this script, it will prompt you to visit a URL to authorize
access to your Box account. After you authorize access, it will error, this
is a bit of a horrid hack. On the error page you will see a URL with
`code=` in it. Copy the code and paste it into the prompt. This will
generate an access token and refresh token, which will be used to
authenticate the Box client. Now the script will run and generate the
_data_ids.yaml file in the current directory.
"""

import os
import re

import yaml

try:
    from boxsdk import Client, OAuth2
except ImportError:
    raise ImportError(
        "The Box SDK is not installed. Please install it with "
        "`pip install boxsdk`."
    )


def _categorise_links(filepath: str) -> str:
    """Categorise the links based on the filename.

    This function is used to categorise the links based on the filename.
    It is used to generate the _data_ids.yaml file, which will be updated
    in new releases as and when new data is added to the database.

    Args:
        filepath (str): The path to the file on Box.

    Returns:
        str: The category of the link.
    """
    if re.search(r"^production_grids", filepath):
        return "ProductionGrids"
    elif re.search(r"^test_data", filepath):
        return "TestData"
    elif re.search(r"^dust_data", filepath):
        return "DustData"
    elif re.search(r"^instruments", filepath):
        return "InstrumentData"
    elif re.search(r"^generation_inputs", filepath):
        return "GenerationData"
    else:
        raise ValueError(
            f"Unknown category for file {filepath}. Please check the "
            "filename and try again."
        )


def _get_files_recursive(folder, client, path=""):
    """Get all the files in a folder recursively.

    Args:
        folder (boxsdk.object.folder.Folder): The folder to get the files from.
        client (boxsdk.client.Client): The Box client to use.
        path (str): The path to the folder.

    Returns:
        list: A list of tuples containing the file object and the path to the
            file.
    """
    all_files = []
    for item in folder.get_items():
        if item.type == "folder":
            all_files.extend(
                _get_files_recursive(
                    client.folder(item.id), client, path + item.name + "/"
                )
            )
        elif item.type == "file":
            all_files.append((item, path))
    return all_files


def _update_box_links_database():
    """Update the _data_ids.yaml database of downloads.

    This function is used for maintenance of the package and the data
    available for download from Box. It is used to generate the
    _data_ids.yaml file, which will be updated in new releases as and when
    new data is added to the database.
    """
    # Get the developer token and secret from the environment variables
    CLIENT_ID = os.getenv("SYNTH_BOX_ID", None)
    CLIENT_SECRET = os.getenv("SYNTH_BOX_SECRET", None)

    # Define the shared folder URL from Box (this is the top level folder)
    SHARED_FOLDER_URL = (
        "https://sussex.box.com/s/a48dk93irkp5bj13zc6xoco5o6phat4j"
    )

    # Define the name of the yaml file we will write to
    OUTPUT_YAML = "_data_ids_.yml"

    # Check if the environment variables are set
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "Missing CLIENT_ID or CLIENT_SECRET in environment variables."
        )

    # Authenticate with Box using the ID and secret
    oauth = OAuth2(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    # Get an access token by a backdoor method...
    auth_url, csrf_token = oauth.get_authorization_url("http://localhost")
    print(f"\n➡️  Visit this URL to authorize access:\n{auth_url}")
    auth_code = input("🔑 Paste the authorization code here: ").strip()
    access_token, refresh_token = oauth.authenticate(auth_code)

    # Create a Box client using the access token
    client = Client(oauth)

    # Get the shared folder ID from the URL
    shared_item = client.get_shared_item(SHARED_FOLDER_URL)
    folder_id = shared_item.id

    # Get all the files in the folder recursively
    all_files = _get_files_recursive(client.folder(folder_id), client)

    output = {
        "TestData": {},
        "DustData": {},
        "InstrumentData": {},
        "GenerationData": {},
        "ProductionGrids": {},
    }

    for file_obj, subfolder in all_files:
        file = client.file(file_obj.id).get()
        print(f"🔗 Processing: {file.name}")

        # Skip the development directory
        if "development" in subfolder:
            print("❌ Skipping development directory")
            continue

        # Skip the README files
        if re.search(r"^README".lower(), file.name.lower()):
            print("❌ Skipping README file")
            continue

        if not file.shared_link:
            file = client.file(file.id).update_info(
                data={
                    "shared_link": {
                        "access": "open",
                        "permissions": {"can_download": True},
                    }
                }
            )

        direct_url = file.shared_link.get("download_url") or None
        category = _categorise_links(subfolder + file.name)

        output[category][file.name] = {
            "file": file.name,
            "direct_link": direct_url,
        }

    # Write the ouput to a yaml file
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(output, f, default_flow_style=False)
