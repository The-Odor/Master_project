using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform[] objectsToFollow; // List of objects to follow
    [Range(0,100)]
    public float minDistance = 5f; // Slider for adjusting minimum distance
    public float smoothSpeed = 0.125f; // Smoothing factor for camera movement
    public Vector3 offset = new Vector3(0, 2, -10); // Offset from the object

    private Transform target; // Current target object
    private int objectIndex = 0;

    void Start()
    {
        // You could put in a GameObject.FindObjectsOfType() 
        // function here to look for a common script in your 
        // robots instead of putting it in manually

        // Check if there are objects to follow
        if (objectsToFollow.Length == 0)
        {
            Debug.LogWarning("No objects to follow.");
            return;
        }
        
        // Select the first object to follow
        target = objectsToFollow[objectIndex];
    }
    void Update()
        {
            // Update the target object based on key presses
            if (Input.GetKeyDown(KeyCode.Q))
            {
                // Handle case where index turns negative
                if(objectIndex == 0)
                    objectIndex = objectsToFollow.Length - 1;

                target = objectsToFollow[objectIndex];
            }
            if (Input.GetKeyDown(KeyCode.E))
            {
                // Handle overflow case
                if(objectIndex == objectsToFollow.Length - 1)
                    objectIndex = 0;

                target = objectsToFollow[objectIndex];
            }

            // Calculate desired position based on target's position and offset
            Vector3 desiredPosition = target.position + offset;

            // Ensure camera maintains minimum distance from the target
            float distance = Vector3.Distance(
                transform.position, 
                desiredPosition
            );
            if (distance < minDistance)
            {
                // Calculate new position with minimum distance
                Vector3 direction = (
                    transform.position - desiredPosition).normalized;
                desiredPosition = target.position + direction * minDistance;
            }

            // Smoothly move the camera towards the desired position
            Vector3 smoothedPosition = Vector3.Lerp(
                transform.position, desiredPosition, 
                smoothSpeed * Time.deltaTime
            );
            transform.position = smoothedPosition;

            // Make the camera look at the target object
            transform.LookAt(target);
        }
    }