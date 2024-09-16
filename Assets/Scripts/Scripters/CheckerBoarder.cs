using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckerBoarder : MonoBehaviour {
    MeshRenderer meshRenderer;
    Material material;
    Texture2D texture;
    [SerializeField] float width = 10.0f;
    // Start is called before the first frame update
    void Start() {
        meshRenderer = GetComponent<MeshRenderer>();
        material = meshRenderer.material;
        texture = new Texture2D(256, 256, TextureFormat.RGBA32, true, true);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Point;
        material.SetTexture("_BaseMap", texture);
        createCheckerboard();
    }

    void createCheckerboard() {
        for (int y=0; y<texture.height; y++) {
            for (int x=0; x<texture.width; x++) {
                texture.SetPixel(
                    x, y,
                    evaluateCheckerboardPixel(x, y)
                );
            }
        }
        texture.Apply();
    }

    Color evaluateCheckerboardPixel(int x, int y) {
        float valueX = (x % (width * 2.0f)) / (width * 2.0f);
        float valueY = (y % (width * 2.0f)) / (width * 2.0f);
        
        // float value = ((x < 0.5) ^ (y < 0.5)) ? 1 : 0;

        int vX = 1;
        if (valueX < 0.5f) {vX = 0;}
        
        int vY = 1;
        if (valueY < 0.5f) {vY = 0;}

        float value = 1;
        if (vX == vY) {value = 0;}

        return new Color(value, value, value, 1.0f);
    }
}
