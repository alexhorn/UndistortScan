# UndistortScan
UndistortScan undistorts scans from bad scanners.

My HP Deskjet 2540 distorts the image when scanning, so I wrote this to fix my scanned documents.

# Installation
```
git clone https://github.com/alexhorn/UndistortScan.git
cd UndistortScan
pip install -r requirements.txt
```

# Usage
Print reference.pdf and scan it at 200 dpi. Save the scan as `scanned.png`.

As an example, scan a document and save it as `document.png`. After you run `python undistort.py --input document.png --output fixed-document.png` `fixed-document.png` should contain an undistorted image of your document.
