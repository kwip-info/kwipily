import XCTest
@testable import KwipMLCore

final class MatrixExampleTests:XCTestCase {
    func testPreparedExampleKeepsBranchesReadableWithoutChangingComputation()throws {
        let document=try WorkflowExamples.preparedMatrixClassifier()
        XCTAssertLessThanOrEqual(document.nodes.map(\.x).max() ?? 0,1900)
        XCTAssertLessThanOrEqual(document.nodes.map(\.y).max() ?? 0,500)
        let nodes=Dictionary(uniqueKeysWithValues:document.nodes.map {($0.address,$0)})
        for (index,node) in document.nodes.enumerated() {
            for other in document.nodes.dropFirst(index+1) {
                XCTAssertTrue(abs(node.x-other.x)>=200 || abs(node.y-other.y)>=130,"Overlapping cards: \(node.name), \(other.name)")
            }
        }
        for edge in document.edges {
            XCTAssertLessThan(try XCTUnwrap(nodes[edge.from.address]).x,try XCTUnwrap(nodes[edge.to.address]).x,"Connections should advance left to right")
        }
        let original=try WorkflowExamples.preparedClassifier(projectID:document.projectID,device:"auto",model:ExampleFlows.matrixClassifier)
        XCTAssertEqual(document.edges,original.edges)
        XCTAssertEqual(document.nodes.map(\.payload),original.nodes.map(\.payload))
        XCTAssertEqual(document.nodes.map(\.address),original.nodes.map(\.address))
    }
    func testBranchedExampleCompilesAndRoundTrips()throws {
        var registry=try ComponentRegistry.builtIn()
        registry.components.removeAll {$0.id == "matmul"}
        registry.components.append(MatrixMultiplyComponent.definition)
        let graph=ExampleFlows.matrixClassifier
        let validation=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(validation.isValid,"\(validation.issues)")
        XCTAssertEqual(validation.tensorLayouts["product"],try BatchedTensorLayout(sample:[2,3,2]))
        XCTAssertEqual(validation.widths["flat"],12)
        XCTAssertEqual(graph.nodes.filter {$0.type == "input"}.count,1)
        XCTAssertEqual(Set(graph.edges.filter {$0.target == "product"}.map(\.targetPort)),["a","b"])
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workflow=try WorkflowExamples.preparedMatrixClassifier(device:"cpu")
        let plan=try WorkflowCompiler.compile(workflow,train:WorkflowExamples.trainAddress,modelRegistry:registry,
            dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.model.graph.nodes.filter {$0.type == "matmul"}.count,1)
        XCTAssertEqual(plan.settings.device,"cpu")
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)),workflow)
    }
    func testExampleTrainingResumeAndPortableWeights()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Local Python runtime unavailable")}
        let temp=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:temp)}
        var registry=try ComponentRegistry.builtIn()
        registry.components.removeAll {$0.id == "matmul"};registry.components.append(MatrixMultiplyComponent.definition)
        try JSONEncoder().encode(registry).write(to:temp.appendingPathComponent("registry.json"))
        try JSONEncoder().encode(ExampleFlows.matrixClassifier).write(to:temp.appendingPathComponent("graph.json"))
        let script="""
        import importlib.util,json,sys,torch
        from pathlib import Path
        from runner.engine import run,GraphModel,load_data
        folder=Path(sys.argv[1]);graph=json.loads((folder/'graph.json').read_text())
        registry=folder/'registry.json'
        options=dict(registry=registry,device='cpu',batch_size=32,sample_limit=128,learning_rate=.01)
        run(graph,folder/'full',epochs=4,**options)
        run(graph,folder/'first',epochs=2,**options)
        run(graph,folder/'resumed',epochs=4,resume_checkpoint=folder/'first'/'checkpoint.pt',**options)
        full=torch.load(folder/'full'/'checkpoint.pt',weights_only=True)
        resumed=torch.load(folder/'resumed'/'checkpoint.pt',weights_only=True)
        for key,value in full['modelState'].items():
            torch.testing.assert_close(value,resumed['modelState'][key],rtol=0,atol=0)
        manifest=json.loads((folder/'resumed'/'manifest.json').read_text())
        assert manifest['status']=='completed' and manifest['lineage']['mode']=='exactResume'
        assert len(json.loads((folder/'resumed'/'metrics.json').read_text()))==4
        spec=importlib.util.spec_from_file_location('exported',folder/'resumed'/'model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        exported=module.Model();exported.load_state_dict(resumed['modelState'],strict=True)
        compiled=GraphModel(graph,registry);compiled.load_state_dict(resumed['modelState'],strict=True)
        for count in [1,2,7]:
            inputs=load_data()[0][2][:count]
            with torch.no_grad():torch.testing.assert_close(exported(inputs),compiled(inputs),rtol=0,atol=0)
        run(graph,folder/'fine',epochs=1,finetune_checkpoint=folder/'resumed'/'checkpoint.pt',seed=19,**options)
        fine=json.loads((folder/'fine'/'manifest.json').read_text())
        assert fine['status']=='completed' and fine['lineage']['mode']=='weightsOnlyFineTune'
        print('Matrix example: exact resume, portable predictions and weights-only fine-tuning passed')
        """
        let process=Process(),pipe=Pipe();process.executableURL=python;process.currentDirectoryURL=root
        process.arguments=["-c",script,temp.path];process.standardOutput=pipe;process.standardError=pipe
        try process.run();let output=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,String(decoding:output,as:UTF8.self))
    }

}
