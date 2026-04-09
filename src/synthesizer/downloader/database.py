"""Generate the `_data_ids.yml` database of downloadable Box assets.

This module is used for package maintenance rather than normal user-facing
workflows. It walks the shared Box folder used by `synthesizer`, ensures files
have public download links, and writes the resulting `_data_ids.yml` file in
the current directory.

This script requires the latest Box Python SDK. Install it with
`pip install "boxsdk>=10"`. The v10+ SDK exposes the generated
`box_sdk_gen` package, which this script uses for OAuth authentication. You
will also need to set the environment variables `SYNTH_BOX_ID` and
`SYNTH_BOX_SECRET` to the client ID and secret of your Box application.

You must then create a Box application at
https://developer.box.com/console/apps (assuming you have a Box account with
edit permissions to the folder). Create an OAuth 2.0 authentication app and
set a Redirect URI to match `SYNTH_BOX_REDIRECT_URI` (default:
"http://127.0.0.1:8080/callback"). The application also needs write access so
that shared links can be created.

When you run this script it will open the Box authorization page in your
browser, start a temporary local callback server, exchange the returned
auth code for an access token, and then update the `_data_ids.yml` database
and create that file in the current directory.

To update the Synthesizer database this script should be run in the
`src/synthesizer/downloader` directory, and the resulting `_data_ids.yml` file
should be committed to the repository.
"""

import json
import os
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import yaml

try:
    from box_sdk_gen import BoxOAuth, GetAuthorizeUrlOptions, OAuthConfig
except ImportError as exc:
    raise ImportError(
        "The latest Box SDK is not installed. Please install it with "
        '`pip install "boxsdk>=10"`.'
    ) from exc


# Define constants for the Box API and shared folder URL.
BOX_API_URL = "https://api.box.com/2.0"
SHARED_FOLDER_URL = "https://sussex.box.com/s/a48dk93irkp5bj13zc6xoco5o6phat4j"
OUTPUT_YAML = "_data_ids.yml"


def _categorise_links(filepath: str) -> str:
    """Categorise a Box filepath into a downloader database section.

    Args:
        filepath (str): Relative path to a Box file, including subdirectories.

    Returns:
        str: The top-level `_data_ids.yml` category for the file.
    """
    # Match the file path against the known downloader categories.
    if re.search(r"^production_grids", filepath):
        return "ProductionGrids"
    if re.search(r"^test_data", filepath):
        return "TestData"
    if re.search(r"^dust_data", filepath):
        return "DustData"
    if re.search(r"^instruments", filepath):
        return "InstrumentData"
    if re.search(r"^generation_inputs", filepath):
        return "GenerationData"
    if re.search(r"^synference", filepath):
        return "SynferenceData"

    raise ValueError(
        f"Unknown category for file {filepath}. Please check the filename "
        "and try again."
    )


def _get_auth_code_via_callback(
    oauth: BoxOAuth, redirect_uri: str, timeout: int = 300
) -> str:
    """Get a Box authorization code through a temporary local callback.

    Args:
        oauth (BoxOAuth): Configured Box OAuth helper used to build the auth
            URL and exchange the returned code later.
        redirect_uri (str): Local callback URI registered in the Box app
            configuration.
        timeout (int): Maximum number of seconds to wait for the callback.

    Returns:
        str: The authorization code returned by Box after user approval.
    """
    # Validate the redirect URI before starting the local callback server.
    parsed_uri = urlparse(redirect_uri)
    if parsed_uri.scheme != "http":
        raise ValueError(
            "SYNTH_BOX_REDIRECT_URI must use http for the local callback "
            "server."
        )
    if parsed_uri.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "SYNTH_BOX_REDIRECT_URI must point to localhost or 127.0.0.1."
        )
    if not parsed_uri.port:
        raise ValueError(
            "SYNTH_BOX_REDIRECT_URI must include an explicit port."
        )

    # Build the authorization URL and capture the expected state token.
    auth_url = oauth.get_authorize_url(
        options=GetAuthorizeUrlOptions(redirect_uri=redirect_uri)
    )
    expected_state = parse_qs(urlparse(auth_url).query).get("state", [None])[0]
    callback_state = {"code": None, "error": None}
    callback_received = Event()

    class _CallbackHandler(BaseHTTPRequestHandler):
        """Handle the single OAuth callback request from Box.

        Args:
            *args: Positional arguments forwarded by `HTTPServer`.
            **kwargs: Keyword arguments forwarded by `HTTPServer`.

        Returns:
            None: The handler writes the HTTP response directly.
        """

        def do_GET(self):
            """Process the OAuth callback request from Box.

            Args:
                self (_CallbackHandler): Active request handler instance.

            Returns:
                None: The response is written directly to the socket.
            """
            # Parse the callback query string and validate the CSRF state.
            parsed_path = urlparse(self.path)
            params = parse_qs(parsed_path.query)
            state = params.get("state", [None])[0]

            if expected_state and state != expected_state:
                callback_state["error"] = (
                    "CSRF token mismatch in Box OAuth callback."
                )
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<h1>Authorization failed</h1><p>CSRF token mismatch.</p>"
                )
                callback_received.set()
                return

            if "error" in params:
                callback_state["error"] = params["error"][0]
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<h1>Authorization failed</h1>"
                    b"<p>You can close this window and check the terminal.</p>"
                )
                callback_received.set()
                return

            callback_state["code"] = params.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authorization complete</h1>"
                b"<p>You can close this window and return to the terminal.</p>"
            )
            callback_received.set()

        def log_message(self, format, *args):
            """Suppress default HTTP server logging during local auth.

            Args:
                format (str): Standard library log message format string.
                *args: Arguments interpolated into the format string.

            Returns:
                None: Logging is intentionally suppressed.
            """
            # Keep the terminal output focused on the downloader workflow.
            return

    # Start a short-lived localhost server to receive the OAuth callback.
    server = HTTPServer(
        (parsed_uri.hostname, parsed_uri.port), _CallbackHandler
    )
    server.timeout = 1

    # Open the browser and poll the local server until Box redirects back.
    print(f"\nOpening Box authorization page:\n{auth_url}")
    print(f"Waiting for OAuth callback at {redirect_uri}")

    if not webbrowser.open(auth_url):
        print("Could not open a browser automatically. Open the URL above.")

    try:
        while not callback_received.is_set():
            server.handle_request()
            timeout -= 1
            if timeout <= 0:
                raise TimeoutError(
                    "Timed out waiting for the Box OAuth callback."
                )
    finally:
        server.server_close()

    # Convert callback state into either a code or a clear exception.
    if callback_state["error"]:
        raise RuntimeError(
            f"Box authorization failed: {callback_state['error']}"
        )
    if callback_state["code"] is None:
        raise RuntimeError("Box authorization did not return a code.")

    return callback_state["code"]


def _get_access_token(oauth: BoxOAuth) -> str:
    """Extract the active access token from the Box OAuth helper.

    Args:
        oauth (BoxOAuth): Authenticated Box OAuth helper.

    Returns:
        str: Bearer token string for authenticated Box API requests.
    """
    # Handle the different token shapes returned by the generated SDK.
    token = oauth.retrieve_token()
    if isinstance(token, str):
        return token

    for attr in ("access_token", "accessToken"):
        value = getattr(token, attr, None)
        if value:
            return value

    raise RuntimeError("Could not retrieve an access token from Box OAuth.")


def _box_api_request(
    access_token: str,
    method: str,
    path: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    payload: dict | None = None,
):
    """Make an authenticated request to the Box REST API.

    Args:
        access_token (str): Bearer token used for Box API authentication.
        method (str): HTTP method to use for the request.
        path (str): API path relative to `BOX_API_URL`.
        params (dict | None): Optional query parameters to append.
        headers (dict | None): Optional extra HTTP headers.
        payload (dict | None): Optional JSON request body.

    Returns:
        dict | list | None: Decoded JSON response body, or `None` when the
        response has no body.
    """
    # Construct the full request URL and normalise multi-value parameters.
    url = f"{BOX_API_URL}{path}"
    if params:
        normalised_params = {}
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                normalised_params[key] = ",".join(str(item) for item in value)
            else:
                normalised_params[key] = value
        url = f"{url}?{urlencode(normalised_params, doseq=True)}"

    # Build the authenticated request headers and optional JSON body.
    request_headers = {
        "Authorization": f"Bearer {access_token}",
    }
    if headers:
        request_headers.update(headers)

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=request_headers, method=method)

    # Execute the request and surface Box errors with their response payloads.
    try:
        with urlopen(request) as response:
            raw_response = response.read()
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Box API request failed for {method} {path}: "
            f"HTTP {exc.code} {error_body}"
        ) from exc

    # Decode JSON responses, while allowing empty success responses.
    if not raw_response:
        return None

    return json.loads(raw_response.decode("utf-8"))


def _get_shared_folder(access_token: str, shared_folder_url: str) -> dict:
    """Resolve the configured shared Box URL to a folder object.

    Args:
        access_token (str): Bearer token used for Box API authentication.
        shared_folder_url (str): Public shared Box folder URL to resolve.

    Returns:
        dict: Folder payload returned by the Box API.
    """
    # Resolve the shared URL into the root folder used by the downloader.
    folder = _box_api_request(
        access_token,
        "GET",
        "/shared_items",
        headers={"boxapi": f"shared_link={shared_folder_url}"},
        params={"fields": ["id", "type", "name"]},
    )

    # Guard against the shared URL unexpectedly resolving to a non-folder.
    if folder["type"] != "folder":
        raise RuntimeError(
            f"Shared URL did not resolve to a folder: {shared_folder_url}"
        )

    return folder


def _get_folder_items(folder_id: str, access_token: str) -> list[dict]:
    """Get all direct children in a Box folder.

    Args:
        folder_id (str): Box folder identifier to inspect.
        access_token (str): Bearer token used for Box API authentication.

    Returns:
        list[dict]: List of folder entries returned across all pages.
    """
    # Walk through paginated folder listings until all entries are collected.
    all_items = []
    offset = 0
    limit = 1000

    while True:
        response = _box_api_request(
            access_token,
            "GET",
            f"/folders/{folder_id}/items",
            params={
                "fields": ["id", "type", "name"],
                "offset": offset,
                "limit": limit,
            },
        )

        # Accumulate one page of entries before advancing the offset.
        entries = response.get("entries", [])
        all_items.extend(entries)
        if len(entries) < limit:
            break
        offset += len(entries)

    return all_items


def _get_files_recursive(
    folder_id: str, access_token: str, path: str = ""
) -> list[tuple[dict, str]]:
    """Get all files contained within a Box folder tree.

    Args:
        folder_id (str): Box folder identifier at the current recursion level.
        access_token (str): Bearer token used for Box API authentication.
        path (str): Relative path prefix accumulated during recursion.

    Returns:
        list[tuple[dict, str]]: Pairs of file payloads and their relative
        parent directory paths.
    """
    # Recurse through nested folders while preserving relative paths.
    all_files = []

    for item in _get_folder_items(folder_id, access_token):
        if item["type"] == "folder":
            all_files.extend(
                _get_files_recursive(
                    item["id"], access_token, path + item["name"] + "/"
                )
            )
        elif item["type"] == "file":
            all_files.append((item, path))

    return all_files


def _get_file(access_token: str, file_id: str) -> dict:
    """Get file metadata needed for shared-link generation.

    Args:
        access_token (str): Bearer token used for Box API authentication.
        file_id (str): Box file identifier to fetch.

    Returns:
        dict: File payload containing the name and shared link fields.
    """
    # Fetch a minimal file payload that is stable across the downloader flow.
    return _box_api_request(
        access_token,
        "GET",
        f"/files/{file_id}",
        params={"fields": ["id", "name", "shared_link"]},
    )


def _ensure_file_has_shared_link(access_token: str, file_id: str) -> dict:
    """Ensure a Box file has an open downloadable shared link.

    Args:
        access_token (str): Bearer token used for Box API authentication.
        file_id (str): Box file identifier to inspect and update.

    Returns:
        dict: Refreshed file payload containing shared-link metadata.
    """
    # Reuse an existing shared link when one is already attached to the file.
    file_info = _get_file(access_token, file_id)
    if file_info.get("shared_link"):
        return file_info

    # Create a shared link that permits direct downloads when missing.
    _box_api_request(
        access_token,
        "PUT",
        f"/files/{file_id}",
        params={"fields": "shared_link,name,id"},
        payload={
            "shared_link": {
                "access": "open",
                "permissions": {"can_download": True},
            }
        },
    )

    # Re-fetch the file because the update response can omit fields we need.
    return _get_file(access_token, file_id)


def _update_box_links_database():
    """Update the local `_data_ids.yml` database of Box downloads.

    Args:
        None.

    Returns:
        None: The database is written to `OUTPUT_YAML` in the current
        directory.
    """
    # Load the Box OAuth configuration from the local environment.
    client_id = os.getenv("SYNTH_BOX_ID")
    client_secret = os.getenv("SYNTH_BOX_SECRET")
    redirect_uri = os.getenv(
        "SYNTH_BOX_REDIRECT_URI", "http://127.0.0.1:8080/callback"
    )

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing SYNTH_BOX_ID or SYNTH_BOX_SECRET in environment "
            "variables."
        )

    # Authenticate with Box and resolve the shared data folder.
    oauth = BoxOAuth(
        OAuthConfig(client_id=client_id, client_secret=client_secret)
    )
    auth_code = _get_auth_code_via_callback(oauth, redirect_uri)
    oauth.get_tokens_authorization_code_grant(auth_code)
    access_token = _get_access_token(oauth)

    shared_folder = _get_shared_folder(access_token, SHARED_FOLDER_URL)
    all_files = _get_files_recursive(shared_folder["id"], access_token)

    # Prepare the output structure expected by the downloader code.
    output = {
        "TestData": {},
        "DustData": {},
        "InstrumentData": {},
        "GenerationData": {},
        "ProductionGrids": {},
        "SynferenceData": {},
    }

    # Walk every file, creating links where needed and writing YAML entries.
    for file_obj, subfolder in all_files:
        file_info = _ensure_file_has_shared_link(access_token, file_obj["id"])
        if "name" not in file_info:
            raise RuntimeError(
                "Box returned an unexpected file payload for "
                f"file id {file_obj['id']}: {file_info}"
            )
        print(f"Processing: {file_info['name']}")

        # Skip files that should not appear in the published database.
        if "development" in subfolder:
            print("Skipping development directory")
            continue

        if re.search(r"^readme", file_info["name"].lower()):
            print("Skipping README file")
            continue

        # Store the direct download URL in the correct category section.
        direct_url = file_info.get("shared_link", {}).get("download_url")
        category = _categorise_links(subfolder + file_info["name"])
        output[category][file_info["name"]] = {
            "file": file_info["name"],
            "direct_link": direct_url,
        }

    # Write the refreshed Box link database to the expected YAML filename.
    with open(OUTPUT_YAML, "w", encoding="utf-8") as out_file:
        yaml.dump(output, out_file, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    _update_box_links_database()
