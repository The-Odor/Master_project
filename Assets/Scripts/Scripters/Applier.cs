using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
// using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

[ExecuteInEditMode]

public class Applier : MonoBehaviour {
    // Components
    public PhysicMaterial physicMaterial;

    // Variables
    float stiffnessFactor = 100000;
    float dampingFactor = 100;
    float forceLimitFactor = 3.402823e38f;
    int networkInputFactor = 12;
    int networkOutputFactor = 2;


    void OnEnable() {

        // Edits behaviour names dependent on morphology
        List<Transform> v1_roots = new List<Transform>();
        foreach (Transform trans in this.transform.GetComponentsInChildren<Transform>()) {
            if (trans.name.EndsWith("_v1")) {
                v1_roots.Add(trans);
            }
        }
        foreach (Transform v1 in v1_roots) {
            foreach (Transform child in v1.GetComponentsInChildren<Transform>()) {
                // Debug.Log(child);
                if (child.GetComponent<BehaviorParameters>() != null) {
                    child.GetComponent<BehaviorParameters>().BehaviorName = v1.name;
                }
            }
        }




        foreach (Transform child in this.transform.GetComponentsInChildren<Transform>()) {
            // if (child.gameObject.name == "Box" && child.GetComponent<BoxCollider>() != null) {
            //     child.GetComponent<BoxCollider>().material = physicMaterial;
            //     // child.GetComponent<BoxCollider>().material = null;
            // }
            // if (
            //     child.GetComponent<ArticulationBody>() == null 
            //     && child.gameObject.name != "Visuals"
            //     && child.gameObject.name != "Scripter"
            //     && child.gameObject.name != "Collisions"
            //     && child.gameObject.name != "Plugins"
            //     && child.gameObject.name != "gecko_v1"
            //     ) {
            //     if (child.GetComponent<BoxCollider>() == null) {
            //         Debug.Log("BoxCollider added");
            //         child.gameObject.AddComponent<BoxCollider>();
            //     }
            //     child.GetComponent<BoxCollider>().material = physicMaterial;
            // //     // DestroyImmediate(child.GetComponent<BoxCollider>());
            // }


                

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
                child.GetComponent<ArticulationBody>().collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
                // Edits BehaviorParameter parameters
                if (child.GetComponent<BehaviorParameters>() != null) {
                    child.GetComponent<BehaviorParameters>().BrainParameters.VectorObservationSize = networkInputFactor;
                    child.GetComponent<BehaviorParameters>().BrainParameters.ActionSpec = new ActionSpec(networkOutputFactor, null);
                    child.GetComponent<BehaviorParameters>().BehaviorType = BehaviorType.Default;
                }
                // Edits ArticulationBody parameters
                // child.GetComponent<ArticulationBody>().LinearDamping = 0.05;
                // child.GetComponent<ArticulationBody>().JointFriction = 0.05;
                // child.GetComponent<ArticulationBody>().AngularDamping = 0.05;
                // Edits DecisionRequester parameters
                if (child.GetComponent<DecisionRequester>()!=null) {
                    child.GetComponent<DecisionRequester>().DecisionPeriod = 1;
                } 
            }
            // Edits layer for collision properties
            child.gameObject.layer = LayerMask.NameToLayer("RobitElement");
        }
    }
}
