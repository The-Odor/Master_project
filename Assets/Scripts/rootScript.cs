using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class rootScript : Agent
{
    List<Vector3> initialPositions = new List<Vector3>();
    List<Transform> transformsList = new List<Transform>();

    // Start is called before the first frame update
    void Start() {
        MaxStep = 12000;
        transformsList.Add(this.transform);
        initialPositions.Add(this.transform.position);
        foreach (Transform child in this.transform.GetComponentsInChildren<Transform>()) {
            transformsList.Add(child.transform);
            initialPositions.Add(child.transform.position);
        }
        Debug.Log(transformsList.Count);
    }

    public override void OnEpisodeBegin() {
        Debug.Log("Root OnEpisodeBegin called");
        for (int i=0; i<transformsList.Count; i++) {
            transformsList[i].position = initialPositions[i];
            Debug.Log("Successfully attempted to move element");
        }
    }
}
