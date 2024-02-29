using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LegSphericalArticulation : MonoBehaviour
{
    public float torque;
    private ArticulationBody articulation;

    // Start is called before the first frame update
    void Start()
    {
        articulation = GetComponent<ArticulationBody>();
        torque = 1;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // float turn = Input.GetAxis("Horizontal");
        Vector3 trans = Random.insideUnitSphere;
        trans = transform.right; 

        // Debug.Log("product: " + trans * torque);
        // Debug.Log("torque: " + torque);
        // Debug.Log("trans: " + trans);
        articulation.AddTorque(trans * torque);
    }
}
