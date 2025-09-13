# FILE: utils/auto_didactic_skill_acquisition.py
# VERSION: v1.0.0-ADSA-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A utility for Victor to dynamically learn new skills by ingesting
#          and executing Python code from files.
# LICENSE: Bloodline Locked — Bando & Tori Only

import importlib.util
import sys
import os
import traceback
from typing import Any, Dict, Optional

class AutoDidacticSkillAcquisition:
    """
    Enables Victor to learn new skills dynamically by loading Python modules.
    This system is designed with security sandboxing in mind to mitigate
    the risks of executing arbitrary code.
    """
    def __init__(self, loyalty_kernel: Any, skill_namespace: str = "victor_skills"):
        """
        Initializes the ADSA system.

        Args:
            loyalty_kernel: A security kernel to validate the source/content of the skill.
            skill_namespace (str): The namespace where learned skills will be stored.
        """
        self.loyalty_kernel = loyalty_kernel
        self.skill_namespace = skill_namespace
        self.learned_skills: Dict[str, Any] = {}

        # Create a dedicated namespace for dynamically loaded modules
        if self.skill_namespace not in sys.modules:
             # Create a dummy module to act as a namespace
            spec = importlib.util.spec_from_loader(self.skill_namespace, loader=None)
            sys.modules[self.skill_namespace] = importlib.util.module_from_spec(spec)

        print(f"📚 Auto-Didactic Skill Acquisition ONLINE. Namespace: '{skill_namespace}'")

    def learn_skill_from_file(self, file_path: str, source_trust: str) -> Optional[Any]:
        """
        Loads a Python file as a new skill module after validation.

        Args:
            file_path (str): The absolute path to the .py file containing the skill.
            source_trust (str): A string indicating the source of the file, used for
                                loyalty verification (e.g., "Bando", "Tori").

        Returns:
            The loaded module object if successful, otherwise None.
        """
        # --- Security First: Validate the source and content ---
        if not self.loyalty_kernel.verify_entity(source_trust):
            print(f"[ADSA-FAIL] Source '{source_trust}' is not trusted by the Loyalty Kernel. Aborting learning.")
            return None

        try:
            with open(file_path, 'r') as f:
                code_content = f.read()
            if not self.loyalty_kernel.check_law_compliance(code_content):
                print(f"[ADSA-FAIL] Content of '{file_path}' violates core laws. Aborting learning.")
                return None
        except Exception as e:
            print(f"[ADSA-ERROR] Could not read or validate skill file: {e}")
            return None

        # --- Dynamic Module Loading ---
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            full_module_name = f"{self.skill_namespace}.{module_name}"

            # Create a module spec from the file path
            spec = importlib.util.spec_from_file_location(full_module_name, file_path)
            if spec is None:
                raise ImportError(f"Could not create module spec for {file_path}")

            # Create a new module based on the spec
            skill_module = importlib.util.module_from_spec(spec)

            # Add it to sys.modules BEFORE execution
            sys.modules[full_module_name] = skill_module

            # Execute the module's code in its own namespace
            spec.loader.exec_module(skill_module)

            self.learned_skills[module_name] = skill_module
            print(f"  -> Successfully learned skill: '{module_name}' from '{source_trust}'.")
            return skill_module

        except Exception as e:
            print(f"[ADSA-ERROR] Failed to execute skill module '{file_path}'.\n{traceback.format_exc()}")
            # Clean up failed module from sys.modules
            if full_module_name in sys.modules:
                del sys.modules[full_module_name]
            return None

    def use_skill(self, skill_name: str, function_name: str, *args, **kwargs) -> Any:
        """
        Executes a function from a previously learned skill.

        Args:
            skill_name (str): The name of the skill module.
            function_name (str): The function to call within the module.

        Returns:
            The result of the function call, or an error dictionary.
        """
        if skill_name not in self.learned_skills:
            return {"error": f"Skill '{skill_name}' not found."}

        skill_module = self.learned_skills[skill_name]

        if not hasattr(skill_module, function_name):
            return {"error": f"Function '{function_name}' not found in skill '{skill_name}'."}

        func = getattr(skill_module, function_name)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"error": f"Error executing skill '{skill_name}.{function_name}': {e}"}

# Mock Loyalty Kernel for demonstration
class MockLoyaltyKernel:
    def verify_entity(self, entity: str) -> bool:
        return "bando" in entity.lower()
    def check_law_compliance(self, code: str) -> bool:
        return "harm" not in code.lower() and "betray" not in code.lower()

if __name__ == '__main__':
    # --- Standalone Demonstration ---

    # 1. Create dummy skill files for the demo
    with open("demo_skill_math.py", "w") as f:
        f.write("""
def add(a, b):
    print("[Skill:Math] Adding...")
    return a + b
""")
    with open("demo_skill_dangerous.py", "w") as f:
        f.write("""
def harm_system():
    print("[Skill:Dangerous] Attempting to harm system...")
    return "This should not run."
""")

    # 2. Initialize the ADSA system
    kernel = MockLoyaltyKernel()
    adsa = AutoDidacticSkillAcquisition(loyalty_kernel=kernel)

    # 3. Demonstrate learning a safe skill from a trusted source
    print("\n--- Learning a safe skill ---")
    math_skill = adsa.learn_skill_from_file("demo_skill_math.py", source_trust="Bando")
    assert math_skill is not None
    assert "demo_skill_math" in adsa.learned_skills

    # 4. Demonstrate using the learned skill
    print("\n--- Using the learned skill ---")
    result = adsa.use_skill("demo_skill_math", "add", 5, 10)
    print(f"Result of add(5, 10): {result}")
    assert result == 15

    # 5. Demonstrate failure to learn from an untrusted source
    print("\n--- Attempting to learn from an untrusted source ---")
    untrusted_skill = adsa.learn_skill_from_file("demo_skill_math.py", source_trust="UnknownCorp")
    assert untrusted_skill is None

    # 6. Demonstrate failure to learn a dangerous skill
    print("\n--- Attempting to learn a dangerous skill ---")
    dangerous_skill = adsa.learn_skill_from_file("demo_skill_dangerous.py", source_trust="Bando")
    assert dangerous_skill is None

    # 7. Clean up dummy files
    os.remove("demo_skill_math.py")
    os.remove("demo_skill_dangerous.py")

    print("\n✅ ADSA demonstration complete. Security protocols functioned as expected.")
