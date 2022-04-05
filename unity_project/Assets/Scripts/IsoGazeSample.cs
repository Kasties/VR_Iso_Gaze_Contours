using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TensorFlowLite;
using UnityEngine.XR;
using UnityEditor;
using System.Linq;
using System;

[System.Serializable]
public class AFMVectors
{
    public Vector3[] vectorsList;

    public int[] trianglesList;

    private bool flip=true;

    public static AFMVectors CreateFromJSON(TextAsset jsonString)
    {   
        
        return JsonUtility.FromJson<AFMVectors>(jsonString.ToString());
        
    }

    public void Rotate(int rotateValue) {

        if (flip) {

            Array.Reverse(vectorsList);
            flip = false;
        }

        int n = vectorsList.Length;
        Vector3[] copy = new Vector3[n];
        
            for (int i = 0; i<n; ++i)
            {
                copy[(i + n - rotateValue) % n] = vectorsList[i];
            }

        vectorsList = copy;
    }


}

public class IsoGazeSample : MonoBehaviour
{
    [SerializeField] string fileName = "isogaze.tflite";
    [SerializeField] Text outputTextView = null;
    [SerializeField] ComputeShader compute = null;
    
    public Directions D;

    Interpreter interpreter;
    
    bool isProcessing = false;
    float[] inputs = new float[3];
    float[] outputs = new float[42];
    ComputeBuffer inputBuffer;

    //afm
    public TextAsset jsonTextFile;
    private AFMVectors afm;
    [SerializeField] float tresh; // when the threshold is reached t0 for afm = 0 , 1-t0 for the model is 1

    public bool interpolate = true;

    [SerializeField] int rotateValue=0;
    private int rotateValueOld=0;

    public Vector3 angVel;

    System.Text.StringBuilder sb = new System.Text.StringBuilder();

    void Start()
    {
        //inputs[0] = -180f; //angle

        var options = new InterpreterOptions()
        {
            threads = 2,
            useNNAPI = true,
        };
        interpreter = new Interpreter(TensorFlowLite.FileUtil.LoadFile(fileName), options);
        interpreter.ResizeInputTensor(0, new int[] { 1, 3 });
        interpreter.AllocateTensors();

        inputBuffer = new ComputeBuffer(3, sizeof(float));

        D = new Directions(transform);

        getCenterEye();

        inputs[2] = 0.7f;


        afm = AFMVectors.CreateFromJSON(jsonTextFile);
    }

    List<InputDevice> Devices;
    InputDeviceCharacteristics desiredCharacteristics;
    InputDevice CenterEye;

    void getCenterEye(){

        Devices = new List<InputDevice>();
        desiredCharacteristics = InputDeviceCharacteristics.HeadMounted;
        InputDevices.GetDevicesWithCharacteristics(desiredCharacteristics, Devices);

        Debug.Log(string.Format("Device name '{0}' has role '{1}'", Devices[0].name, Devices[0].role.ToString()));

        CenterEye = Devices[0];


    }

    void OnDestroy()
    {
        interpreter?.Dispose();
        inputBuffer?.Dispose();
    }


    private void AngularVelocity(out float angularvelocityangle, out float velocitymagnitude)
    {
        
        if (CenterEye.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out angVel))
        {
            velocitymagnitude = angVel.magnitude;
            angVel = new Vector3(Mathf.Deg2Rad * angVel.y, Mathf.Deg2Rad * -angVel.x, 0f);
            angularvelocityangle = (Vector3.Angle(Vector3.up, angVel) * Mathf.Sign(angVel.x));
        }
        else 
        {
            angVel = Vector3.zero;
            angularvelocityangle = 0f;
            velocitymagnitude = 0f;
        }

    }

    //private void EulerToSpherical(Vector3 euler, out Vector2 direction ) { 
    //    elevation = Mathf.Deg2Rad * euler.x;
    //    heading = Mathf.Deg2Rad * euler.y;
    //    //direction = new Vector3(Mathf.Cos(elevation) * Mathf.Sin(heading), Mathf.Sin(elevation), Mathf.Cos(elevation) * Mathf.Cos(heading));
    //}

    Vector3[] InterpolateModels(Vector3[] afm, Vector3[] model, float t)
    {
        Vector3[] res = new Vector3[model.Length];
        for (int i=0; i< model.Length; i++)
        {
            //t==0 afm              t==1 model
            res[i] = (1-t)*afm[i] + t*model[i];
        }

        return res;
    }

    void Invoke()
    {
        isProcessing = true;

        AngularVelocity(out inputs[0],out inputs[1]);

        //manual input
        //inputs[0] = -180f; //angle
        //inputs[1] = 8.0f; //speed d/s
        //inputs[2] = 0.7f; //percentile


        //pure model output
        float startTime = Time.realtimeSinceStartup;
        interpreter.SetInputTensorData(0, inputs);
        interpreter.Invoke();
        interpreter.GetOutputTensorData(0, outputs);
        
        float duration = Time.realtimeSinceStartup - startTime;
        
        XY angles = ToXY(outputs);

        D.setOrientation(angles.x, angles.y);

        if (interpolate) {

            float t = 1;
            if (angVel.magnitude <= tresh) // if v >=threshold then t = 1 , model will be used otherwise interpolation with afm
            {
                t = angVel.magnitude / tresh; //this division normalise the parameter to something between 0 and 1, closer to 0 is afm
            }

            //afm-model interpolation
            D.DirectionsArray = InterpolateModels(afm.vectorsList, D.DirectionsArray, t); //0 means afm, 1 model
        }

        if (outputTextView != null) { 

            //Debug.Log(outputs);
            sb.Clear();
            sb.AppendLine($"Process time: {duration: 0.00000} sec");
            sb.AppendLine("---");
            sb.AppendLine($"output[0]: {outputs[0]: 0.00000} sec");
            sb.AppendLine("---");
            sb.AppendLine($"inputs[0]: {inputs[0]: 0.00000} sec");
            sb.AppendLine("---");
            sb.AppendLine($"inputs[1]: {inputs[1]: 0.00000} sec");
            sb.AppendLine("---");
            sb.AppendLine($"inputs[2]: {inputs[2]: 0.00000} sec");
            outputTextView.text = sb.ToString();

        }

        
        isProcessing = false;
    }

    struct XY
    {
        public float[] x;
        public float[] y;
    }
    private XY ToXY(float[] list) {

        XY xy = new XY { };
        xy.x = new float[20]; //not the last
        xy.y = new float[20]; //not the last

        for (int i = 0; i < list.Length-2; i++) //not the last
        {
            int index = Mathf.FloorToInt(i/2);
            if (i % 2 != 0) 
                xy.x[index] = list[i];
            else 
                xy.y[index] = list[i];
        }

        return xy;
    }

    void Update()
    {
        Invoke();

        if (Input.GetKeyDown("n"))
        {
            if (inputs[2] <= 1) inputs[2] += 0.1f;
            else inputs[2] = 1f;
        }
        else if (Input.GetKeyDown("m"))
        {
            if (inputs[2] > 0.1f) inputs[2] -= 0.1f;
            else inputs[2] = 0.1f;
        }

        if (rotateValue != rotateValueOld) {

            afm.Rotate(rotateValue);
            rotateValueOld = rotateValue;
        }

    }

    void OnDrawGizmos()
    {   
        Gizmos.color = Color.green;

        if (D==null)  return; 
        foreach (Vector3 direction in D.DirectionsArray)
        {

            Gizmos.DrawRay(transform.position, direction);

        }

        Gizmos.DrawRay(Vector3.zero, angVel * 3000);

        foreach (Vector3 point in D.PointsArray)
        {

            Gizmos.DrawSphere(point * 10, 1f);

        }

        //Gizmos.DrawRay(transform.TransformPoint(Vector3.zero), transform.TransformPoint(Vector3.forward));

        //Gizmos.DrawRay(transform.TransformPoint(Vector3.zero), transform.TransformVector(Vector3.forward * 60));

    }

    public class Directions 
    {

        public Transform parent;

        public Vector3[] DirectionsArray = new Vector3[20];
        public Vector3[] PointsArray = new Vector3[20];


        public Directions(Transform p) {

            parent = p;
        }


        public void setOrientation(float[] x, float[] y)
        {


            for (int i = 0; i < DirectionsArray.Length; i++)
            {
                float horRads = Mathf.Deg2Rad * x[i];
                float verRads = Mathf.Deg2Rad * y[i];

                PointsArray[i] = new Vector3(y[i], x[i], 0f);

                //DirectionsArray[i] = parent.TransformVector(SphericalToCartesian(200, horRads, verRads));
                DirectionsArray[i] = SphericalToCartesian(1, horRads, verRads);
            }

        }

        public static Vector3 SphericalToCartesian(float radius, float polar, float elevation)
        {
            Vector3 outCart = new Vector3();

            float a = radius * Mathf.Cos(elevation);
            outCart.x = radius * Mathf.Sin(elevation);
            outCart.y = a * Mathf.Sin(polar);
            outCart.z = a * Mathf.Cos(polar);

            return outCart;
        }


    }
}

