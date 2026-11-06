


how to run project for MAC OS:


1. chnage directory to your folder
cd dl_karthikeya_waste   


2. create virtual environment:
python3 -m venv .venv

3. activate venv:
source .venv/bin/activate


4. python -m pip install --upgrade pip


5. install all dependecies
pip install -r requirements.txt

6. run backend
uvicorn backend.main:app --reload


7. run frontend in new terminal
cd dl_karthikeya_waste/frontend

python3 -m http.server 5500

8. open http://127.0.0.1:5500/  





how to run on winows 


1. Navigate into the project folder:
cd dl_karthikeya_waste

2. Create a virtual environment:
python -m venv .venv

3. Activate the virtual environment:
.venv\Scripts\activate

4. Upgrade pip:
python -m pip install --upgrade pip

5. Install all dependencies:
pip install -r requirements.txt

6. Start the backend:
uvicorn backend.main:app --reload

Backend runs at:
http://127.0.0.1:8000  (open and test backend)

Keep this terminal running.

7. Open a SECOND PowerShell window.
Navigate to the frontend folder:

cd dl_karthikeya_waste\frontend


8. Start the frontend:
python -m http.server 5500


9. Open your browser and visit:
http://127.0.0.1:5500
