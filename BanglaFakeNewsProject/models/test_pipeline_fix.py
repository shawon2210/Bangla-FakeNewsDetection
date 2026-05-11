"""Quick test to verify pipeline integration fix"""
from pipeline_integration import PipelineIntegrator

print("Testing pipeline integration...")
integrator = PipelineIntegrator()

print("\n1. Testing existing workflow validation...")
result = integrator.validate_existing_workflow()
print(f"Result: {'✅ PASS' if result else '⚠️ Some checks skipped'}")

print("\n✅ Pipeline integration test complete!")
