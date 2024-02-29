using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]

public class Visualizer : MonoBehaviour
{
    public Material robot_material;
    public Mesh robot_mesh;
    public float penis;

    // Start is called before the first frame update
    void OnEnable()
    {
        // Debug.Log("Started");
        foreach (Transform tran in this.transform.GetComponentsInChildren<Transform>()){
            // BoxCollider collider = t.GetComponent<BoxCollider>;
            // Debug.Log("transform found");
            if (tran.GetComponent<BoxCollider>()!=null){
                MeshRenderer mr;
                MeshFilter mf;
                if (tran.GetComponent<MeshRenderer>()!=null || tran.GetComponent<MeshFilter>()!=null){
                    mr = tran.gameObject.GetComponent<MeshRenderer>();
                    mf = tran.gameObject.GetComponent<MeshFilter>();
                }
                else {
                    mr = tran.gameObject.AddComponent<MeshRenderer>();
                    mf = tran.gameObject.AddComponent<MeshFilter>();
                    Debug.Log("Components added");
                }
                mr.material = robot_material;
                mf.sharedMesh = robot_mesh;        
            }
        }
    }

}
