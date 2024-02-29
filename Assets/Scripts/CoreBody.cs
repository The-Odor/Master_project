using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CoreBody : MonoBehaviour
{
    Vector3 scale = new Vector3(5,1,5);

    // MeshFilter meshFilter;

    Collider m_collider;
    Vector3 m_size;
    Rigidbody rb;
    ArticulationBody AB;
    Vector3 position;

    // Start is called before the first frame update
    void Start()
    {

        //// Try out approach with Vertices. Didn't impact anything afaik
        // meshFilter = GetComponent<MeshFilter>();
        // Mesh mesh = meshFilter.mesh;    
        // Vector3[] vertices = mesh.vertices;
        // Debug.Log("Length: " + vertices.Length);

        // for (int i=0; i<vertices.Length; i++){
        //     float vX = vertices[i].x;
        //     float vY = vertices[i].y;
        //     float vZ = vertices[i].z;
        //     vertices[i] = new Vector3(vX*scale.x, vY*scale.y, vZ*scale.z);
        // }

        // for (int i=0; i<vertices.Length; i++){
        //     Debug.Log(vertices[i]);
        // }

        // m_collider = GetComponent<Collider>();
        // m_size = m_collider.bounds.size;
        // Debug.Log(m_size);
        // transform.isKinematic = true;
        rb = GetComponent<Rigidbody>();
        AB = GetComponent<ArticulationBody>();
        position = new Vector3(0,0,-10);
        
    }

    // Update is called once per frame
    void Update()
    {
        float movement = 5f * Time.deltaTime;
        // transform.Translate(new Vector3(1,1,1)*movement);
        // AB.localPosition(new Vector3(1,1,1)*movement);

        // if (Input.GetKey(KeyCode.W)) 
        if (Input.GetKey(KeyCode.W))
        {
            AB.velocity = (position - transform.position).normalized;
            // Debug.Log(Input.GetKeyDown(KeyCode.W));     
        }
    }
}
