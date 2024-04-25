using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
// using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

[ExecuteInEditMode]

public class Applier : MonoBehaviour {
    float stiffnessFactor = 1;
    float dampingFactor = 1;
    float forceLimitFactor = 10f;
    int networkInputFactor = 12;
    int networkOutputFactor = 2;


    void OnEnable() {
        foreach (Transform child in this.transform.GetComponentsInChildren<Transform>()) {
            if (child.GetComponent<ArticulationBody>() != null) {
                if (!child.name.EndsWith("_root")) {
                    // Adds Controller
                    if (child.GetComponent<MultimorphAgent>()==null) {
                        child.gameObject.AddComponent<MultimorphAgent>();
                    }
                    // Adds Decision Requester
                    if (child.GetComponent<DecisionRequester>()==null) {
                        child.gameObject.AddComponent<DecisionRequester>();
                    }
                }
                // Edits xDrive parameters    
                var drive = child.GetComponent<ArticulationBody>().xDrive;
                drive.stiffness = stiffnessFactor;
                drive.damping = dampingFactor;
                drive.forceLimit = forceLimitFactor;
                child.GetComponent<ArticulationBody>().xDrive = drive;
                // Edits Articulationcontroller parameters
                child.GetComponent<ArticulationBody>().collisionDetectionMode = CollisionDetectionMode.Continuous;
                // Edits BehaviorParameter parameters
                if (child.GetComponent<BehaviorParameters>() != null) {
                    child.GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = networkInputFactor;
                    child.GetComponent<BehaviorParameters>().BrainParameters.ActionSpec = new ActionSpec(networkOutputFactor, null);
                    child.GetComponent<BehaviorParameters>().BehaviorType = BehaviorType.Default;
                }
            }
        }
    }
}
