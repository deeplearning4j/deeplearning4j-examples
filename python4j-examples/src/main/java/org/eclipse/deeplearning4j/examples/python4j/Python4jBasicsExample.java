/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.eclipse.deeplearning4j.examples.python4j;

import org.nd4j.python4j.PythonExecutioner;
import org.nd4j.python4j.PythonGIL;
import org.nd4j.python4j.PythonObject;
import org.nd4j.python4j.PythonTypes;
import org.nd4j.python4j.PythonVariable;
import org.nd4j.python4j.PythonVariables;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Python4j Basics Example
 *
 * This example demonstrates how to use Python4j to run Python code from Java.
 * Python4j is a Python execution library that allows Java programs to:
 *   - Execute Python code strings directly
 *   - Pass typed Java variables into the Python interpreter
 *   - Retrieve typed results back from the Python interpreter
 *   - Work safely in multi-threaded environments via the Global Interpreter Lock (GIL)
 *
 * Python4j uses CPython (the standard Python runtime) embedded in the JVM via
 * JavaCPP. The PythonGIL must be acquired before any Python API call to ensure
 * thread safety -- always use try-with-resources on PythonGIL.lock().
 *
 * Key Python4j classes:
 *   PythonExecutioner  - main entry point: exec(), getVariable(), execAndReturnAllVariables()
 *   PythonGIL          - Global Interpreter Lock; must be held for every Python call
 *   PythonVariable<T>  - a named, typed variable (Java side)
 *   PythonVariables    - a collection of PythonVariable objects (inputs or outputs)
 *   PythonTypes        - factory/type-descriptor for INT, FLOAT, STR, BOOL, LIST, DICT, BYTES
 *   PythonObject       - low-level wrapper around a CPython PyObject*
 */
public class Python4jBasicsExample {

    public static void main(String[] args) throws Exception {

        // -------------------------------------------------------------------------
        // 1. Execute simple Python code with no inputs or outputs
        // -------------------------------------------------------------------------
        System.out.println("=== 1. Executing simple Python code ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            // exec() runs an arbitrary Python code string in the interpreter.
            // Any variables created by the script persist in the Python global namespace
            // for the lifetime of this PythonGIL session.
            PythonExecutioner.exec("x = 1 + 2");
            PythonExecutioner.exec("message = 'Hello from Python4j!'");
            PythonExecutioner.exec("squared = x ** 2");
            System.out.println("Executed: x = 1 + 2, message = 'Hello from Python4j!', squared = x ** 2");
        }

        // -------------------------------------------------------------------------
        // 2. Read Python variables back into Java
        // -------------------------------------------------------------------------
        System.out.println("\n=== 2. Reading variables back from Python ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            // getVariable() fetches a named variable from the Python namespace and
            // converts it to the Java type described by the PythonTypes argument.
            PythonVariable<Integer> xVar = PythonExecutioner.getVariable("x", PythonTypes.INT);
            PythonVariable<String> msgVar = PythonExecutioner.getVariable("message", PythonTypes.STR);
            PythonVariable<Integer> sqVar = PythonExecutioner.getVariable("squared", PythonTypes.INT);

            System.out.println("x       = " + xVar.getValue());    // 3
            System.out.println("message = " + msgVar.getValue());  // Hello from Python4j!
            System.out.println("squared = " + sqVar.getValue());   // 9
        }

        // -------------------------------------------------------------------------
        // 3. Pass Java variables into Python as typed inputs
        // -------------------------------------------------------------------------
        System.out.println("\n=== 3. Passing Java variables into Python ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            // Build an input PythonVariables collection from PythonVariable objects.
            // PythonVariable<T>(name, PythonTypes.XXX, javaValue) wraps any Java value.
            PythonVariables inputs = new PythonVariables();
            inputs.addInt("a", 10L);
            inputs.addFloat("b", 3.14);
            inputs.addStr("label", "result");

            // Declare the output variable we want back -- no value yet, just the name and type.
            PythonVariables outputs = new PythonVariables();
            outputs.addFloat("product");

            String code = "product = a * b";
            PythonExecutioner.exec(code, inputs, outputs);

            double product = outputs.getFloatValue("product");
            System.out.println("a=" + 10 + ", b=" + 3.14 + " => a * b = " + product);
        }

        // -------------------------------------------------------------------------
        // 4. Pass LIST and DICT variables
        // -------------------------------------------------------------------------
        System.out.println("\n=== 4. Passing LIST and DICT variables ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            // Python lists map to java.util.List, dicts to java.util.Map.
            List<Object> numbers = Arrays.asList(1, 2, 3, 4, 5);

            PythonVariables inputs = new PythonVariables();
            inputs.addList("numbers", numbers);

            PythonVariables outputs = new PythonVariables();
            outputs.addFloat("total");
            outputs.addInt("count");

            String code =
                "total = float(sum(numbers))\n" +
                "count = len(numbers)";

            PythonExecutioner.exec(code, inputs, outputs);

            System.out.println("numbers = " + numbers);
            System.out.println("sum     = " + outputs.getFloatValue("total"));
            System.out.println("count   = " + outputs.getIntValue("count"));
        }

        // -------------------------------------------------------------------------
        // 5. Pass DICT variables (Python dict <-> Java Map)
        // -------------------------------------------------------------------------
        System.out.println("\n=== 5. Passing DICT (Map) variables ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            Map<Object, Object> config = new HashMap<>();
            config.put("lr", 0.001);
            config.put("epochs", 10);
            config.put("batch_size", 32);

            PythonVariables inputs = new PythonVariables();
            inputs.addDict("config", config);

            PythonVariables outputs = new PythonVariables();
            outputs.addStr("summary");

            String code =
                "summary = f\"lr={config['lr']}, epochs={config['epochs']}, batch={config['batch_size']}\"";

            PythonExecutioner.exec(code, inputs, outputs);
            System.out.println("config dict = " + config);
            System.out.println("summary     = " + outputs.getStrValue("summary"));
        }

        // -------------------------------------------------------------------------
        // 6. Execute a multi-line Python script
        // -------------------------------------------------------------------------
        System.out.println("\n=== 6. Multi-line Python script ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            String script =
                "import math\n" +
                "\n" +
                "def fibonacci(n):\n" +
                "    a, b = 0, 1\n" +
                "    seq = []\n" +
                "    for _ in range(n):\n" +
                "        seq.append(a)\n" +
                "        a, b = b, a + b\n" +
                "    return seq\n" +
                "\n" +
                "fib_seq = fibonacci(10)\n" +
                "fib_sum = sum(fib_seq)\n" +
                "pi_approx = math.pi\n";

            PythonExecutioner.exec(script);

            // Fetch results individually after the script runs.
            PythonVariable<Long> fibSumVar = PythonExecutioner.getVariable("fib_sum", PythonTypes.INT);
            PythonVariable<Double> piVar = PythonExecutioner.getVariable("pi_approx", PythonTypes.FLOAT);
            PythonVariable<List> fibSeqVar = PythonExecutioner.getVariable("fib_seq", PythonTypes.LIST);

            System.out.println("Fibonacci sequence (10 terms): " + fibSeqVar.getValue());
            System.out.println("Sum of Fibonacci sequence    : " + fibSumVar.getValue());
            System.out.println("math.pi                      : " + piVar.getValue());
        }

        // -------------------------------------------------------------------------
        // 7. Boolean and conditional Python code
        // -------------------------------------------------------------------------
        System.out.println("\n=== 7. Boolean variables and conditionals ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonVariables inputs = new PythonVariables();
            inputs.addInt("threshold", 50L);
            inputs.addInt("value", 75L);

            PythonVariables outputs = new PythonVariables();
            outputs.addInt("is_above");   // Python bool comes back as int (1/0)
            outputs.addStr("verdict");

            String code =
                "is_above = int(value > threshold)\n" +
                "verdict = 'PASS' if value > threshold else 'FAIL'";

            PythonExecutioner.exec(code, inputs, outputs);

            System.out.println("value=" + 75 + ", threshold=" + 50);
            System.out.println("is_above = " + (outputs.getIntValue("is_above") == 1));
            System.out.println("verdict  = " + outputs.getStrValue("verdict"));
        }

        // -------------------------------------------------------------------------
        // 8. execAndReturnAllVariables -- inspect the whole Python namespace
        // -------------------------------------------------------------------------
        System.out.println("\n=== 8. execAndReturnAllVariables ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            // This convenience method runs the code and returns every variable created
            // by the script as a PythonObject map, without needing to declare outputs up front.
            String code =
                "alpha = 1.0\n" +
                "beta  = 2.0\n" +
                "gamma = alpha + beta\n";

            PythonVariables result = PythonExecutioner.execAndReturnAllVariables(code);
            System.out.println("Variables returned by execAndReturnAllVariables:");

            // PythonVariables.getVariables() gives the raw variable map.
            for (String name : result.getVariables()) {
                System.out.println("  " + name + " = " + result.getValue(name));
            }
        }

        // -------------------------------------------------------------------------
        // 9. Using PythonObject for low-level access
        // -------------------------------------------------------------------------
        System.out.println("\n=== 9. Low-level PythonObject access ===");

        try (PythonGIL gil = PythonGIL.lock()) {
            PythonExecutioner.exec("greeting = 'Hello, Java!'");

            // PythonObject wraps a raw CPython object reference. It supports
            // attribute access, item access, calling, and conversion to Java types.
            PythonObject obj = PythonExecutioner.getVariable("greeting", PythonTypes.STR).getPythonObject();
            System.out.println("PythonObject.toString() = " + obj.toString());
            System.out.println("str length via Python   = " + obj.attr("__len__").call().toInt());
        }

        System.out.println("\nPython4jBasicsExample complete.");
    }
}
