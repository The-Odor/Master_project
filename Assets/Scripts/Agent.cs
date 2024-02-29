using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent : MonoBehaviour
{
    List<Transform> segments = new List<Transform>();

    // Start is called before the first frame update
    void Start(){
        // Prepares list containing all joints in 
        foreach (Transform tran in this.transform.GetComponentsInChildren<Transform>()){
            if (tran.GetComponent<ArticulationBody>()!=null){
                segments.Add(tran);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // /// <summary>
    // /// Collect vector observations from the environment
    // /// </summary>
    // /// <param name="sensor">The vector sensor</param>
    // public override void CollectObservations(VectorSensor sensor) {
    //     sensor.AddObservation(1);
    //     sensor.AddObservation(2);
    // }

    // /// <summary>
    // /// Called when an action is received from the neural network
    // /// </summary>
    // /// <param name="">The actions to take</param>
    // public override void OnActionReceived(float[] vectorAction) {

    // }


}
