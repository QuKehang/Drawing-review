<div align="center">
    
# Tesseract: Drawing Information Extraction and Comparison


</div>

## Introduction
In bridge structure design drawings, the position of drawing numbers and names is relatively fixed. This study first manually selects a sample drawing to mark its information coordinates. Based on this, an automated cropping algorithm is developed to traverse the entire set of drawings, accurately extracting the drawing numbers and names on each page according to the preset coordinates, and saving the results to a text file. During the drawing analysis process, the text file records the drawing number and name information for each page, with data divided into three parts: page number, drawing number, and drawing name. Page numbers are used to locate the corresponding parts in the Excel comparison file for comparative analysis. To improve the accuracy of the comparison, only Chinese characters are extracted from the drawing name information, and only numbers are extracted from the drawing number information. The drawing number strings are segmented to accurately determine whether they are within the range specified by the directory. Finally, the comparison results of drawing numbers and names with the directory are output for each page to ensure the standardization and completeness of the design drawings. 

## Software installation

clone this repository and install the dependencies:
```bash
conda create -n paddle python=3.9
conda activate tesseract
pip install .
```

Tested with:
- [pytesseract](https://github.com/madmaze/pytesseract) == 0.3.0
- [opencv-python](https://github.com/opencv-python) == 4.8.1.78
- [pillow](https://github.com/pillow) == 10.1.0
- [numpy](https://github.com/numpy) == 1.24.4
- [pandas]( https://github.com/pandas-dev) == 2.0.3
- [openpyxl](  https://github.com/jmcnamara/XlsxWriter) == 3.1.5



## Getting Started

### Data

What you should do before you start:
#### main1:
- `选择refPts(坐标)文件夹` : The folder containing the coordinates obtained manually.
- `选择输入的原始图片文件夹` : The folder containing the images to be detected.
- `选择保存处理后的图片文件夹` : The folder to save the processed images.
- `选择存储box信息的文件夹` :  The folder for storing box information.
- `txt文件路径` : The folder for saving the final txt result files.

#### main2:
- `TXT文件路径` :  TXT file to be compared.
- `excel文件路径` : Excel file for comparison.


## Obtaining Fixed Coordinates


Manually write code to select the position to obtain information on the original image and obtain the coordinates folder.

```bash
python coordinate_picking.py 
```


## Collecting Drawing Information


Traverse the folder of drawings containing each PNG file, use the coordinate file to structurally extract the required information, and save it as a txt file.

```bash
python main1.py 
```


## Information Comparison


During the drawing information extraction process, each page of the drawing mainly contains three sets of information: page number, drawing name, and drawing number. In the information comparison, the page number is used to ensure accurate comparison and analysis with the corresponding parts in the xlsx comparison file. To improve recognition accuracy, only Chinese characters are retained for the comparison of "drawing name" information, and only numerical characters are retained for "drawing number" information. When segmenting the drawing number information in the txt file and the xlsx file, the first two numbers must be compared to see if they are consistent, and then the subsequent information must be judged for a match.

```bash
python main2.py 
```






