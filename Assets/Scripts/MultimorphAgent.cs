using System.Collections;
using System.Collections.Generic;
// using System.Diagnostics;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using System; // Temporary for testing
using System.Threading; // Is this the only way to introduce a delay??

// cd Master_project
// venv\Scripts\activate
// mlagents-learn Config\MultimorphLocomotion.yaml --run-id=throwaway --force

/// <summary>
/// A locomotive duplicable Machine Learning Agent
/// </summary>
public class MultimorphAgent : Agent {
    // List<Transform> segments = new List<Transform>();
    double TAU;
    
    // Network output factors
    float angleChangeFactor = 1e1f;
    // float velocityChangeFactor = 1e3f;
    // float angleScalingFactor = 1.8e2f;
    // float velocityScalingFactor = 4e2f;
    
    // Robot structural factors
    float stiffnessFactor = 8;
    float dampingFactor = 1;
    float forceLimitFactor = 3.402823e3f;
    float angleUpperLimit = 60;
    float angleLowerLimit = -60;

    // Movement variables
    float currentTargetAngle;
    // float currentTargetVelocity;

    // Training variables
    float rewardFactor = 1e-0f;
    bool disqualified = false;
    float disqualificationHeight = 10;
    float disqualificationPunishment = -1e3f;

    // Other variables
    ArticulationBody articulationBody; 
    Vector3 startPosition;
    Quaternion startRotation;
    ArticulationBody rootBody;
    Vector3 rootStartPosition;
    Quaternion rootStartRotation;
    DateTime timeEpisodeStart;
    GameObject UIText;

    /// <summary>
    /// Initialize the agent
    /// </summary>
    public override void Initialize()
    {
        // TODO
        return;
    }

    // Start is called before the first frame update
    void Start() {
        articulationBody = this.transform.GetComponent<ArticulationBody>();

        // TAU = 2*Math.PI; // It's a much superior variable, fight me on it

        var drive = articulationBody.xDrive;
        drive.lowerLimit = angleLowerLimit;
        drive.upperLimit = angleUpperLimit;
        articulationBody.xDrive = drive;

        MaxStep = 6400;

        currentTargetAngle = 0f;
        // currentTargetVelocity = 0f;

        startPosition = this.transform.position;
        startRotation = this.transform.rotation;

        if (this.transform.parent.name.EndsWith("_v1_root") && this.transform.name.EndsWith("_v1_root_link0")) {
            // startPosition = (this.transform.parent.parent.position);
            Transform rootObject = this.transform.parent;
            rootBody = rootObject.GetComponent<ArticulationBody>();
            rootStartPosition = rootObject.transform.position;
            rootStartRotation = rootObject.transform.rotation;
        }


        UIText = GameObject.Find("UI/Canvas/WindowVersion/Text");
        if (UIText is null) {throw new Exception("UI not found");}
        

        // Fault: These properties are documented, but don't exist
        // articulationBody.SetDriveLimits(drive, -180, 180);
        // articulationBody.SetDriveDamping();
        // articulationBody.SetDriveStiffness();
 
        // Fault: Gets overriden in runtime by something unknown
        // drive.stiffness = 1;
        // drive.damping = 1;
        // drive.forceLimit = 1e38f;
    }

    /// <summary>
    /// Reset the agent when an episode begins
    /// </summary>
    public override void OnEpisodeBegin() {
        // Debug.Log("New episode beginning");
        // Reset location for bot
        // startPosition = transform.position;
        // transform.position = startPosition;
        // Debug.Log("Episode Begun, set position to" + startPosition);
        // this.transform.position = new Vector3(100,100,100);
        if (this.transform.parent.name.EndsWith("_v1_root") && this.transform.name.EndsWith("0")) {
            this.transform.parent.parent.position = startPosition;
        }

        timeEpisodeStart = DateTime.Now;

        // DateTime startTime = DateTime.Now;
        // Debug.Log("DateTime: " + (DateTime.Now - startTime).TotalSeconds);
        // while ((DateTime.Now - startTime).TotalSeconds < 5) {

        //     if (this.transform.name == "gecko_v1_root_link0") {
        //         Debug.Log($"Position: {this.transform.position}; startPosition: {startPosition}" + "\nDateTime within loop: " + (DateTime.Now - startTime).TotalSeconds);
        //     }
        //     // articulationBody.velocity = new Vector3(1e30f,1e30f,1e30f);
        //     articulationBody.velocity = this.transform.position - startPosition;
        //     Thread.Sleep(1000);
        //     // return;
        // }

        // if (this.transform.parent.name.EndsWith("_v1_root") && this.transform.name.EndsWith("_v1_root_link0")) {
        //     rootBody.TeleportRoot(rootStartPosition, rootStartRotation);
        // }

        // articulationBody.velocity = new Vector3(0,0,0);


        // For now, since I can't reset position, I'm just gonna pretend
        // startPosition = transform.position;
        

        // Debug.Log("New episode begun");
    }

    // public void FixedUpdate() {
    public override void Heuristic(in ActionBuffers actionsOut) {
        bool printLog = false;
        float switchVal;
        float timeNow;
        int numberOfOptions = 2;
        int durationOfTorque = 10; // seconds

        if (printLog) {Debug.Log("Heuristic called");}
        timeNow = (float)DateTime.Now.Second + (float)DateTime.Now.Minute*60;
        switchVal = timeNow % (numberOfOptions*durationOfTorque);

        //Cycles through modes every durationOfTorque seconds to test function
        int action;
        switch (switchVal) {
            case var switchVal_ when (0*durationOfTorque <= switchVal_ && switchVal_  <= 1*durationOfTorque):
                if (printLog) {Debug.Log("Pushing positive");}
                action =  1;
                break;
            case var switchVal_ when (1*durationOfTorque < switchVal_ && switchVal_  <= 2*durationOfTorque):
                if (printLog) {Debug.Log("Pushing negative");}
                action = -1;
                break;
            default:
                if (printLog) {Debug.Log("Heuristic defaulting to no motion");}
                action =  0;
                break;
        }

        RotateAction(action, action);
        if (printLog) {Debug.Log("Heuristic Completed for values of " + (action) + ", " + (action));}
    }

    /// <summary>
    /// Collect vector observations from the environment
    /// </summary>
    /// <param name="sensor">The vector sensor</param>
    public override void CollectObservations(VectorSensor sensor) {
        // 12 total inputs


        // Parent input (1DoF -> 2 input)       
        if (this.transform.parent.name.EndsWith("_v1_root")) {
            // if parent is root add empty reading
            sensor.AddObservation(0.0f);
            sensor.AddObservation(0.0f);
        } else {
            ArticulationBody articulationParent = this.transform.parent.GetComponent<ArticulationBody>();
            if (float.IsNaN(articulationParent.jointPosition[0])) 
                {sensor.AddObservation(0.0f); Debug.Log("NaN observation detected");} 
            else {sensor.AddObservation(articulationParent.jointPosition[0]);}
            if (float.IsNaN(articulationParent.jointVelocity[0])) 
                {sensor.AddObservation(0.0f); Debug.Log("NaN observation detected");} 
            else {sensor.AddObservation(articulationParent.jointVelocity[0]);}
            // sensor.AddObservation(0.0f);
        }
        // Own rotation (1 DoF -> 2 input)
        if (float.IsNaN(articulationBody.jointPosition[0])) 
            {sensor.AddObservation(0.0f); Debug.Log("NaN observation detected");} 
        else {sensor.AddObservation(articulationBody.jointPosition[0]);}
        if (float.IsNaN(articulationBody.jointVelocity[0])) 
            {sensor.AddObservation(0.0f); Debug.Log("NaN observation detected");} 
        else {sensor.AddObservation(articulationBody.jointVelocity[0]);}
        // sensor.AddObservation(0.0f);

        // Child rotations (1DoF * 4 -> 8 inputs)
        int childCount = 0;
        foreach (Transform child in this.transform) {
            if (child.GetComponent<ArticulationBody>() != null) {
                ArticulationBody articulationChild = child.GetComponent<ArticulationBody>();
                if (float.IsNaN(articulationChild.jointPosition[0])) 
                    {sensor.AddObservation(0.0f); Debug.Log("NaN observation detected");} 
                else {sensor.AddObservation(articulationChild.jointPosition[0]);}
                if (float.IsNaN(articulationChild.jointVelocity[0])) 
                    {sensor.AddObservation(0.0f); Debug.Log("NaN observation detected");} 
                else {sensor.AddObservation(articulationChild.jointVelocity[0]);}
                // sensor.AddObservation(0.0f);
                childCount++;
            }
        } 
        // If there are fewer than 4 children, add empty readings
        sensor.AddObservation(new float[(4-childCount)*2]);


        // Reward is cumulated over time as horizontal displacement from origin
        // Cumulative reward rewards high initial motion.
        float reward;
        // string statement = "";
        if (disqualified || transform.position[1] > disqualificationHeight){
            disqualified = true;
            reward = disqualificationPunishment;
            // statement = ", because it is DISQUALIFIED";
        } else {
            reward = rewardFactor*(float)Math.Sqrt(
                Math.Pow(transform.position[0] - startPosition[0], 2)
              + Math.Pow(transform.position[2] - startPosition[2], 2)
                );
        }
        // Debug.Log("Reward added: " + reward + statement);
        if (!float.IsNaN(reward)) {AddReward(reward);}
        else {Debug.Log("Reward somehow became a NaN??");}

        // UI Update section
        if (this.transform.name.EndsWith("v1_root_link0")) {
            TMPro.TextMeshProUGUI mText = UIText.GetComponent<TMPro.TextMeshProUGUI>();
            Vector3 pos = transform.position;
            float dist = (float)Math.Sqrt(Math.Pow(pos.x,2) + Math.Pow(pos.y,2));
            string name = this.transform.name;
            name = name.Substring(0, name.IndexOf("v1_root_link0"));
            if (this.transform.name == "gecko_v1_root_link0") {
                mText.text = ""+name + dist;
            } else {
                mText.text = mText.text + "\n" + name + dist;
            }
        }
    }

    /// <summary>
    /// Called when an action is received from the neural network
    /// </summary>
    /// <param name="">The actions to take</param>
    public override void OnActionReceived(ActionBuffers actions) {
        // 2 total outputs
        // if ((DateTime.Now - timeEpisodeStart).TotalSeconds < 1 && false) {
        //     var drive = articulationBody.xDrive;
        //     drive.stiffness = stiffnessFactor;
        //     drive.damping = dampingFactor;
        //     drive.forceLimit = forceLimitFactor;
        //     drive.targetVelocity = -drive.target;
        //     drive.target = 0;
        //     articulationBody.xDrive = drive;
        // } else {
            if (!(float.IsNaN(actions.ContinuousActions[0])) && (!float.IsNaN(actions.ContinuousActions[1]))){
                RotateAction(actions.ContinuousActions[0], actions.ContinuousActions[1]);
            } else {Debug.Log("Mooooom, the network output a NaaaaN!!");}
        // }
    }

    // Trying to copy from the ArmController, using some of their helpers
    private void RotateAction(float newTargetAngleNormalized, float newTargetVelocityNormalized) {
        // A bigger motion is made faster
        // Scaled linearly by angleScalingFactor and the difference
        // between current angle and target angle
        // float newTargetAngle = newTargetAngleNormalized * angleScalingFactor;
        // currentTargetAngle = Mathf.MoveTowards(
        //     currentTargetAngle, 
        //     newTargetAngle, 
        //     Math.Abs(newTargetAngle - currentTargetAngle)*angleChangeFactor//*Time.fixedDeltaTime
        // );
        // float newTargetVelocity = newTargetVelocityNormalized * velocityScalingFactor;
        // currentTargetVelocity = Mathf.MoveTowards(
        //     currentTargetVelocity, 
        //     newTargetVelocity, 
        //     Math.Abs(newTargetVelocity - currentTargetVelocity)*velocityChangeFactor//*Time.fixedDeltaTime
        // );

        // Until we think we need the actual smoothing, we're going to just make it work like this
        currentTargetAngle = newTargetAngleNormalized * angleChangeFactor;
        // currentTargetVelocity = newTargetVelocityNormalized * velocityChangeFactor;

        var drive = articulationBody.xDrive;
        drive.stiffness = stiffnessFactor;
        drive.damping = dampingFactor;
        drive.forceLimit = forceLimitFactor;

        drive.target = currentTargetAngle;// + articulationBody.jointPosition[0];
        // drive.targetVelocity = currentTargetVelocity;
        // drive.targetVelocity = 0;
        
        articulationBody.xDrive = drive;
        
        // Debug.Log("drive.target/newTargetVelocity set to: " + articulationBody.xDrive.target + "; " + articulationBody.xdrive.targetVelocity);

        // Fault: Only works on root
        // articulationBody.SetDriveTargets(new List<float> {target});
        
        // Fault: Will spin entire body
        // articulationBody.AddRelativeTorque(primaryAxisRotation * new Vector3(1,0,0));

        // Physics.Simulate(Time.fixedDeltaTime);
    }
}

