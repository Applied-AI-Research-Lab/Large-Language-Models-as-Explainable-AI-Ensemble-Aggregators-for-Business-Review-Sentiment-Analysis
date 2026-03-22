# bash 1_download_yelp_review.sh
#!/usr/bin/env bash
set -euo pipefail

ZIP_URL="https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADS_DIR="$ROOT_DIR/downloads"
ZIP_PATH="$DOWNLOADS_DIR/Yelp-JSON.zip"
EXTRACT_DIR="$DOWNLOADS_DIR"
UNTAR_DIR="$EXTRACT_DIR/Yelp JSON"
TAR_PATH="$UNTAR_DIR/yelp_dataset.tar"
SOURCE_JSON="$EXTRACT_DIR/Yelp JSON/yelp_academic_dataset_review.json"
TARGET_JSON="$ROOT_DIR/yelp_academic_dataset_review.json"

mkdir -p "$DOWNLOADS_DIR"

echo "Downloading Yelp dataset archive..."
curl --fail --location --retry 3 --retry-delay 2 --user-agent "$USER_AGENT" "$ZIP_URL" -o "$ZIP_PATH"

echo "Validating downloaded archive..."
if ! unzip -tqq "$ZIP_PATH" >/dev/null 2>&1; then
  echo "Downloaded file is not a valid zip archive: $ZIP_PATH"
  echo "This usually means the remote server blocked the request or returned an error page."
  exit 1
fi

echo "Unzipping into $DOWNLOADS_DIR ..."
unzip -o "$ZIP_PATH" -d "$EXTRACT_DIR"

if [[ ! -f "$TAR_PATH" ]]; then
  echo "Expected tar file not found after unzip: $TAR_PATH"
  exit 1
fi

echo "Extracting $TAR_PATH ..."
tar -xf "$TAR_PATH" -C "$UNTAR_DIR"

if [[ ! -f "$SOURCE_JSON" ]]; then
  echo "Expected file not found: $SOURCE_JSON"
  exit 1
fi

echo "Copying review JSON to project root..."
cp "$SOURCE_JSON" "$TARGET_JSON"

echo "Done."
echo "- Downloaded zip: $ZIP_PATH"
echo "- Extracted under: $EXTRACT_DIR"
echo "- Extracted tar: $TAR_PATH"
echo "- Copied review file: $TARGET_JSON"