using System.Collections;
using System.Collections.Generic;
using UnityEngine;



[CreateAssetMenu(menuName = "Singletons/ScriptableTest")]
public class ScriptableTest : SingletonScriptableObject<ScriptableTest>
{
    [SerializeField]
    private int _gameSettings=1;

    public static int GameSettings
    {
        get {
            Debug.Log("15");
            Debug.Log(Instance.ToString());
            Debug.Log("16");
            Debug.Log(Instance._gameSettings.ToString());
            Debug.Log("17");
            return Instance._gameSettings;
        }

    }
}
