import XCTest
@testable import KwipMLCore

final class GELUExampleTests:XCTestCase {
    func testBothModesArePortableValidatedSequentialExamples()throws {
        let registry=try ComponentRegistry.builtIn()
        for mode in [false,true] {
            let graph=ExampleFlows.geluClassifier(approximateTanh:mode)
            XCTAssertEqual(graph.schemaVersion,2)
            XCTAssertEqual(graph.nodes.map(\.type),["input","linear","gelu","linear","output"])
            XCTAssertEqual(graph.nodes.first(where:{$0.type=="gelu"})?.config["approximate_tanh"],.bool(mode))
            XCTAssertTrue(GraphValidator.validate(graph,registry:registry).isValid)
            let execution=try XCTUnwrap(graph.execution)
            XCTAssertEqual(execution.epochs,100);XCTAssertEqual(execution.device,"auto")
            XCTAssertNil(execution.datasetCSV);XCTAssertNil(execution.preparedTraining);XCTAssertNil(execution.compositeSource)
            XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:JSONEncoder().encode(graph)),graph)
        }
        XCTAssertEqual(ExampleFlows.parallelClassifier.nodes.filter{$0.type=="gelu"}.count,0)
    }

    func testActualBundledFactoryGraphsTrainAndExportForBothModes()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Local Python runtime unavailable")}
        let temp=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:temp)}
        let script="""
        import importlib.util,json,sys,torch
        from pathlib import Path
        from runner.engine import run,GraphModel,load_data
        graph=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);settings=graph['execution']
        run(graph,out,epochs=settings['epochs'],seed=settings['seed'],device='cpu',batch_size=settings['batchSize'],split_seed=settings['splitSeed'],validation_percent=settings['validationPercent'],learning_rate=settings['learningRate'])
        checkpoint=torch.load(out/'checkpoint.pt',weights_only=True)
        spec=importlib.util.spec_from_file_location('exported',out/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        exported=module.Model();exported.load_state_dict(checkpoint['modelState'],strict=True)
        compiled=GraphModel(graph);compiled.load_state_dict(checkpoint['modelState'],strict=True)
        inputs=load_data()[0][2][:8]
        with torch.no_grad():torch.testing.assert_close(exported(inputs),compiled(inputs),rtol=0,atol=0)
        manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['graph']==graph
        assert len(json.loads((out/'metrics.json').read_text()))==settings['epochs']
        print(json.dumps({'status':'completed','mode':graph['nodes'][2]['config']['approximate_tanh'],'epochs':settings['epochs'],'validationLoss':manifest['evaluation']['validationLoss']}))
        """
        for mode in [false,true] {
            let input=temp.appendingPathComponent("\(mode).json"),output=temp.appendingPathComponent("run-\(mode)")
            try JSONEncoder().encode(ExampleFlows.geluClassifier(approximateTanh:mode)).write(to:input)
            let process=Process(),pipe=Pipe();process.executableURL=python;process.currentDirectoryURL=root
            process.arguments=["-c",script,input.path,output.path];process.standardOutput=pipe;process.standardError=pipe
            try process.run();let bytes=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus,0,String(decoding:bytes,as:UTF8.self))
        }
    }
}
