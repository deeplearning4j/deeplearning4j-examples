package com.example.androidDl4jClassifier;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;


public class MainActivity extends AppCompatActivity {

    private static MainActivity instance;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        instance = this;
        setContentView(R.layout.activity_main);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        ScatterView view = findViewById(R.id.id_scatterview);
        switch (item.getItemId()) {
            case R.id.action_linear:
                view.showDataset("linear_data_train.csv");
                return true;

            case R.id.action_moon:
                view.showDataset("moon_data_train.csv");
                return true;

            case R.id.action_saturn:
                view.showDataset("saturn_data_train.csv");
                return true;

            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                return super.onOptionsItemSelected(item);

        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    public static MainActivity getInstance(){
        return instance;
    }

}
