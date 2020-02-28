# FACE DETECTION
Detect face, age and sex from camera

## HOW TO RUN DEMO
1. Install Anaconda for Window 10 64bit using Python 3
    - https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe

2. Open **Anaconda Prompt** and run the following comand:
    
   ```bash
   cd <project-folder>
   conda env create -f environment.yml
   conda activate FACE-DETECTION
   python demo.py
   ```
## HOW TO PACK DEMO.PY TO DEMO.EXE

1. Install Pyinstaller in Anaconda env
   ```
   conda install -c conda-forge pyinstaller
   ```

2. Create demo.exe using Pyinstaller
    ```
    pyinstaller --onefile demo.py
    ```
3. Copy `demo.exe` from `dist` folder into project root folder
4. Click on `demo.exe` to run the program

## Happy Coding