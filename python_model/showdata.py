import pickle

try:
    with open('/home/mar/ai/VR_Iso_Gaze_Contours/python_model/dataset9/360_VR_gaze.pkl', 'rb') as pkl_file:
        loaded_data = pickle.load(pkl_file)
        print("Loaded data:", loaded_data)
        print(loaded_data.sample(5))  # Sample 5 random rows



except FileNotFoundError:
    print("File not found.")
except pickle.PickleError as e:
    print("Error loading the PKL file:", e)
