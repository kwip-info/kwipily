import XCTest
@testable import KwipMLCore

final class LayerNormExampleTests:XCTestCase {
    func testShippedDefinitionAndPreparedWorkflowScopes() throws {
        let registry=try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type:"layer_norm"),LayerNormComponent.definition)
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for whole in [false,true] {
            let document=try WorkflowExamples.preparedLayerNormClassifier(device:"cpu",normalizeWholeSample:whole)
            let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,
                dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(plan.settings.device,"cpu")
            XCTAssertEqual(plan.settings.epochs,3)
            XCTAssertEqual(plan.sourceAddresses,[WorkflowExamples.sourceAddress])
            XCTAssertEqual(plan.model.graph.nodes.first(where:{$0.type == "layer_norm"})?.config["normalized_shape"],.array(whole ? [2,3]:[3]))
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        }
    }
    func testExampleIsPortableAndPreservesSampleShape()throws {
        var registry=try ComponentRegistry.builtIn()
        registry.components.removeAll {$0.id == "layer_norm"}; registry.components.append(LayerNormComponent.definition)
        for mode in [false,true] {
            let graph=ExampleFlows.layerNormClassifier(normalizeWholeSample:mode)
            XCTAssertEqual(graph.schemaVersion,2)
            XCTAssertEqual(graph.nodes.map(\.type),["input","linear","reshape","layer_norm","gelu","flatten","linear","output"])
            XCTAssertEqual(graph.nodes.first(where:{$0.type=="layer_norm"})?.config["normalized_shape"],.array(mode ? [2,3] : [3]))
            XCTAssertTrue(GraphValidator.validate(graph,registry:registry).isValid)
            let execution=try XCTUnwrap(graph.execution)
            XCTAssertEqual(execution.epochs,20);XCTAssertEqual(execution.device,"auto")
            XCTAssertNil(execution.datasetCSV);XCTAssertNil(execution.preparedTraining);XCTAssertNil(execution.compositeSource)
            XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:JSONEncoder().encode(graph)),graph)
        }
    }

    func testFactoryGraphsTrainAndExportForBothNormalizationScopes()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Local Python runtime unavailable")}
        let temp=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:temp)}
        var registry=try ComponentRegistry.builtIn();registry.components.removeAll {$0.id == "layer_norm"}; registry.components.append(LayerNormComponent.definition)
        let registryURL=temp.appendingPathComponent("registry.json")
        try JSONEncoder().encode(registry).write(to:registryURL)
        let script="""
        import importlib.util,json,sys,torch
        from pathlib import Path
        from runner.engine import run,GraphModel,load_data
        graph=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);settings=graph['execution']
        run(graph,out,registry=Path(sys.argv[3]),epochs=settings['epochs'],seed=settings['seed'],device='cpu',batch_size=settings['batchSize'],sample_limit=settings['sampleLimit'],split_seed=settings['splitSeed'],validation_percent=settings['validationPercent'],learning_rate=settings['learningRate'])
        checkpoint=torch.load(out/'checkpoint.pt',weights_only=True)
        spec=importlib.util.spec_from_file_location('exported',out/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        exported=module.Model();exported.load_state_dict(checkpoint['modelState'],strict=True)
        compiled=GraphModel(graph,Path(sys.argv[3]));compiled.load_state_dict(checkpoint['modelState'],strict=True)
        inputs=load_data()[0][2][:8]
        with torch.no_grad():torch.testing.assert_close(exported(inputs),compiled(inputs),rtol=0,atol=0)
        manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['graph']==graph
        assert len(json.loads((out/'metrics.json').read_text()))==settings['epochs']
        print(json.dumps({'status':'completed','mode':graph['nodes'][3]['config']['normalized_shape'],'epochs':settings['epochs'],'validationLoss':manifest['evaluation']['validationLoss']}))
        """
        for mode in [false,true] {
            let input=temp.appendingPathComponent("\(mode).json"),output=temp.appendingPathComponent("run-\(mode)")
            try JSONEncoder().encode(ExampleFlows.layerNormClassifier(normalizeWholeSample:mode)).write(to:input)
            let process=Process(),pipe=Pipe();process.executableURL=python;process.currentDirectoryURL=root
            process.arguments=["-c",script,input.path,output.path,registryURL.path];process.standardOutput=pipe;process.standardError=pipe
            try process.run();let bytes=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus,0,String(decoding:bytes,as:UTF8.self))
        }
    }
}
