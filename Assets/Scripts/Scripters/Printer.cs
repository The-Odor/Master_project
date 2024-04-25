using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using Unity.MLAgents.Policies;
using UnityEngine;

[ExecuteInEditMode]

// Used to print information from different modules
public class Printer : MonoBehaviour {

    string parentName;
    ArticulationBody articule;
    string largestChildFlockName;
    string largestParent;
    int childCount;
    void OnEnable() {
        // int largestChildCount = 0;
        foreach (Transform child in this.transform.GetComponentsInChildren<Transform>()){
            // ### Acquire name ###
            if (child.name.EndsWith("_v1")) {
                parentName = child.name;
                // Debug.Log("parent found");
            } else {
            // ### Acquire jointType ###
                articule = child.GetComponent<ArticulationBody>();
                if (articule!=null){
                    // Debug.Log(parentName);
                    // Debug.Log(articule.jointType); // They are all revolute
                    // Debug.Log(child.rotation);
                }
            }
            // Debug.Log(child.childCount + "" + child.name);


            // ### Acquire largest number of children for any BrticulationBody
            // childCount = 0;
            // if (child.GetComponent<ArticulationBody>() != null){
            //     foreach (Transform directChild in child) {
            //         if (directChild.GetComponent<ArticulationBody>() != null) {
            //             childCount++;
            //         }
            //     }
            //     if ((childCount > largestChildCount)) {
            //         largestChildCount = childCount;
            //         largestChildFlockName = child.name;
            //         largestParent = parentName;
            //     }
            // }
            // Debug.Log(largestChildFlockName + " of " + largestParent + ": " + largestChildCount);
            // The largest amount of children is 4

            // ### Acquire all names transform names ###
            // foreach (Transform child in transform) {
            //     Debug.Log(child.name);
            // }

            // ### Acquire DoF for all ArticulationBodies ###
            // All have 1 DoF except for the roots, who have 0
            // if (child.GetComponent<ArticulationBody>() != null) {
            //     int DoF = child.GetComponent<ArticulationBody>().jointPosition.dofCount;
            //     Debug.Log("DoF: " + DoF);
            //     if (DoF == 0) {Debug.Log(child.name);}
            // }            

            // ### Acquire all components of gameobject ###
            // if (child.name == "ant_v1_root_link0") {
            //     Component[] components = child.GetComponents(typeof(Component));
            //     foreach(Component component in components) {
            //         Debug.Log(component.ToString());
            //     }
            // }

            // -------------- SINGLE SEGMENT SECTION --------------
            if (child.name == "ant_v1_root_link0") {
                

                // ### Acquire names of BehaviourParameters fields&properties ###
                // Type type = child.GetComponent<BehaviorParameters>().GetType();
                // Type type = child.GetComponent<BehaviorParameters>().BrainParameters.GetType();
                // Type type = child.GetComponent<BehaviorParameters>().BrainParameters.ActionSpec.GetType();
                // Type type = child.GetComponent<BehaviorParameters>().BrainParameters.ActionSpec.NumContinuousActions.GetType();
                // Type type = child.GetComponent<BehaviorParameters>().BrainParameters.ActionSpec.NumDiscreteActions.GetType();
                // Type type = child.GetComponent<BehaviorParameters>().BrainParameters.ActionSpec.SumOfDiscreteBranchSizes.GetType();
                // Type type = child.GetComponent<BehaviorParameters>().ObservableAttributeHandling.GetType();
                //
                // FieldInfo[] fields = type.GetFields();
                // PropertyInfo[] props = type.GetProperties();
                // foreach (FieldInfo field in fields) {Debug.Log("field: " + field);}
                // foreach (PropertyInfo prop in props) {Debug.Log("prop: " + prop.ToString());}
                //
                // var obj = child.GetComponent<BehaviorParameters>();
                // Type parentType = null;
                // Type childType = obj.GetType();
                // GetPropertiesAndFields(childType, parentType);
                //
                // Debug.Log(child.GetComponent<BehaviorParameters>().BrainParameters);
                // Debug.Log(child.GetComponent<BehaviorParameters>().BrainParameters.ActionSpec.NumContinuousActions);

                // ### Acquire names of ArticulationBody fields&properties ###
                // Type type = child.GetComponent<ArticulationBody>().collisionDetectionMode.GetType();
                // FieldInfo[] fields = type.GetFields();
                // PropertyInfo[] props = type.GetProperties();
                // foreach (FieldInfo field in fields) {Debug.Log("field: " + field);}
                // foreach (PropertyInfo prop in props) {Debug.Log("prop: " + prop.ToString());}
            }
        }
    }
    void GetPropertiesAndFields(Type type, Type previous) {
        FieldInfo[] fields = type.GetFields();
        foreach (FieldInfo field in fields) {
            Debug.Log("field: " + field + "\nChild of: " + previous);
        }

        PropertyInfo[] props = type.GetProperties();
        foreach (PropertyInfo prop in props) {
            Debug.Log("Prop: " + prop + ", of type: " + prop.GetType() + "\nChild of: " + previous);
            // Module fuckyou = new Module();
            if ((prop.ToString() != "System.Reflection.Module Module") && 
                (prop.ToString() != "System.Reflection.PropertyAttributes Attributes") &&
                (prop.ToString() != "System.Type PropertyType") &&
                (prop.ToString() != "Boolean CanWrite") &&
                (prop.ToString() != "Boolean CanRead")) {
                // Debug.Log("Calling Get for " + prop + ", which allegedly isn't " + typeof(Module));
                Debug.Log("AAAHHHHHHHHHHH " + prop + "; " + prop.ToString());
                GetPropertiesAndFields(prop.GetType(), type);

                // IS THE ISSUE THAT I CALL ON PropertyInfo.GetType()? THAT COULD BE AN ISSUE? THEN HOW DO I DRAW OUT THE ACTUAL PROPERTY?
            }
        }
    }
}
