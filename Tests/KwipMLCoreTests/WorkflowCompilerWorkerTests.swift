import XCTest
import CryptoKit
import Darwin
@testable import KwipMLCore

/// Independently authored current preparation plus the original model route form
/// the oracle; no worker artifacts or expected numerical outputs are synthesized.
final class WorkflowCompilerWorkerTests:XCTestCase {
    private let encoder:JSONEncoder = {let value=JSONEncoder();value.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return value}()
    private func sha(_ data:Data)->String {SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
    private func a(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func independentPreparation()->DataFlowDocument {
        // Explicitly choose the current three-way Partition for this new fixture.
        // It is not an automatic migration of the retired NumPy two-way split.
        let identity=CurrentPracticeConfiguration.columns([])
        let nodes:[DataFlowNode]=[
            .init(id:"source",operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")]),
            .init(id:"cast",operatorID:"kwip.data.columns",config:CurrentPracticeConfiguration.columns([("a","float64"),("b","float64"),("label","int64")])),
            .init(id:"split",operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":17,"validationPercent":20,"testPercent":20,"stratifyColumn":.string("label")]),
            .init(id:"fit",operatorID:"kwip.data.preprocess",config:CurrentPracticeConfiguration.scaledPartitions(["a","b"])),
            .init(id:"training",operatorID:"kwip.data.columns",config:identity),
            .init(id:"validation",operatorID:"kwip.data.columns",config:identity)]
        func e(_ source:String,_ port:String,_ target:String,_ input:String)->DataFlowEdge {.init(from:.init(node:source,port:port),to:.init(node:target,port:input))}
        return .init(nodes:nodes,edges:[e("source","table","cast","table"),e("cast","table","split","table"),e("split","training","fit","training"),e("split","validation","fit","validation"),e("split","test","fit","test"),e("fit","training","training","table"),e("fit","validation","validation","table")])
    }
    private func workflow(preparation:DataFlowDocument,source:ModelAuthoringDocument,registry:DataFlowOperatorRegistry)throws->WorkflowDocument {
        var document=try WorkflowCodec.migrateModel(.blockModel(source),projectID:source.projectID).document
        let data=try WorkflowCodec.migrateData(.init(graph:preparation),projectID:source.projectID,registry:registry).document
        document.nodes += data.nodes;document.edges += data.edges
        document.nodes += [
            .init(address:a(.control,"definition"),payload:.modelDefinition),
            .init(address:a(.control,"mapping"),payload:.trainingData(config:["featureColumns":.array([.string("a"),.string("b")]),"targetColumn":.string("label")])),
            .init(address:a(.control,"setup"),payload:.runSetup(config:["device":.string("cpu"),"epochs":.integer(3),"seed":.integer(13),"batchSize":.integer(8),"learningRate":.number(0.02)])),
            .init(address:a(.control,"train"),payload:.train(config:[:]))]
        func e(_ id:String,_ from:WorkflowDomain,_ node:String,_ port:String,_ to:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
            .init(id:id,from:.init(address:a(from,node),port:port),to:.init(address:a(.control,to),port:input),kind:kind)
        }
        document.edges += [e("workflow-model",.model,"output","definition","definition","source",.modelDefinition),e("workflow-definition",.control,"definition","model","train","model",.modelDefinition),e("workflow-training",.data,"training","table","mapping","training",.table),e("workflow-validation",.data,"validation","table","mapping","validation",.table),e("workflow-mapping",.control,"mapping","data","train","data",.trainingData),e("workflow-setup",.control,"setup","configuration","train","configuration",.runConfiguration)]
        return document
    }
    private struct WorkerResult {let code:Int32;let text:String}
    /// File-backed output avoids pipe deadlock; time and output growth are bounded.
    private func worker(_ python:URL,root:URL,temp:URL,args:[String])throws->WorkerResult {
        let log=temp.appendingPathComponent("worker-\(UUID().uuidString).log")
        _=FileManager.default.createFile(atPath:log.path,contents:nil)
        let handle=try FileHandle(forWritingTo:log);defer{try? handle.close()}
        let process=Process();process.executableURL=python;process.currentDirectoryURL=root
        process.arguments=args;process.standardOutput=handle;process.standardError=handle
        try process.run();let deadline=Date().addingTimeInterval(40)
        var exceeded=false
        while process.isRunning {
            let size=(try? log.resourceValues(forKeys:[.fileSizeKey]).fileSize) ?? 0
            if Date()>deadline || size>2_000_000 {exceeded=true;process.terminate();break}
            Thread.sleep(forTimeInterval:0.025)
        }
        if exceeded {
            let stop=Date().addingTimeInterval(2)
            while process.isRunning && Date()<stop {Thread.sleep(forTimeInterval:0.01)}
            if process.isRunning {_=Darwin.kill(process.processIdentifier,SIGKILL)}
        }
        process.waitUntilExit()
        let reader=try FileHandle(forReadingFrom:log);defer{try? reader.close()}
        let bytes=try reader.read(upToCount:2_000_001) ?? Data()
        guard !exceeded,bytes.count<=2_000_000 else {throw DataFlowContractError("Integration worker exceeded its 40-second or 2 MB output bound.")}
        return .init(code:process.terminationStatus,text:String(decoding:bytes,as:UTF8.self))
    }
    func testCompiledWorkflowMatchesIndependentLegacyPreparationTrainingAndExport()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Optional local Python runtime unavailable; compiler unit tests remain independent.")}
        let temp=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:temp)}
        let dataRegistry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),modelRegistry=try ComponentRegistry.builtIn()
        let baseline=independentPreparation(),settings=ExecutionSettings(device:"cpu",epochs:3,batchSize:8,seed:13,learningRate:0.02)
        var source=CompositeExamples.parallelClassifier;source.name="Compiler worker parity";source.execution=settings
        let authored=ModelAuthoringDocument(projectID:UUID().uuidString,revision:9,source:source)
        let expected=try ModelExecutionSnapshotBuilder.build(.blockModel(authored),registry:modelRegistry)
        let document=try workflow(preparation:baseline,source:authored,registry:dataRegistry)
        let plan=try WorkflowCompiler.compile(document,train:a(.control,"train"),modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        XCTAssertEqual(plan.dataflow,baseline);XCTAssertEqual(plan.model,expected);XCTAssertEqual(plan.settings,settings)
        let csv=temp.appendingPathComponent("fixture.csv"),bindings=temp.appendingPathComponent("bindings.json")
        try (["a,b,label"]+(0..<40).map{"\($0),\($0 * 2 + 1),\($0 % 2)"}).joined(separator:"\n").write(to:csv,atomically:true,encoding:.utf8)
        try encoder.encode(["source":csv.path]).write(to:bindings)
        let main=root.appendingPathComponent("runner/main.py").path
        var preparationOutputs:[URL]=[],modelOutputs:[URL]=[],manifestPins:[String]=[]
        for (index,graph) in [baseline,plan.dataflow].enumerated() {
            let request=temp.appendingPathComponent("prep-\(index).json"),out=temp.appendingPathComponent("prep-\(index)")
            try encoder.encode(graph).write(to:request)
            let result=try worker(python,root:root,temp:temp,args:[main,"--execute-dataflow",request.path,"--source-bindings",bindings.path,"--output",out.path])
            XCTAssertEqual(result.code,0,result.text);guard result.code==0 else{return}
            preparationOutputs.append(out);manifestPins.append(sha(try Data(contentsOf:out.appendingPathComponent("manifest.json"))))
        }
        XCTAssertEqual(manifestPins[0],manifestPins[1],"Identical preparation inputs should preserve graph/partition/state/artifact hashes with cache off")
        func trainArguments(index:Int,output:URL)throws->[String] {
            let snapshot=index==0 ? expected : plan.model
            let graph=temp.appendingPathComponent("model-\(index).json"),provenance=temp.appendingPathComponent("source-\(index).json")
            try encoder.encode(snapshot.graph).write(to:graph)
            let sourceBytes=try encoder.encode(XCTUnwrap(snapshot.provenance));try sourceBytes.write(to:provenance)
            let manifest=preparationOutputs[index].appendingPathComponent("manifest.json")
            let value=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:manifest)) as? [String:Any])
            let digest=try XCTUnwrap(value["graphDigest"] as? String)
            let mapping=index==0 ? ("training","table","validation","table",["a","b"],"label") : (plan.training.node,plan.training.port,plan.validation.node,plan.validation.port,plan.featureColumns,try XCTUnwrap(plan.targetColumn))
            let config=index==0 ? settings : plan.settings
            return [main,"--graph",graph.path,"--output",output.path,"--device",config.device,"--epochs",String(config.epochs),"--seed",String(config.seed),"--batch-size",String(config.batchSize),"--learning-rate",String(config.learningRate),"--prepared-dataflow",manifest.path,"--prepared-manifest-sha256",manifestPins[index],"--prepared-graph-digest",digest,"--prepared-training-node",mapping.0,"--prepared-training-port",mapping.1,"--prepared-validation-node",mapping.2,"--prepared-validation-port",mapping.3,"--feature-columns-json",String(decoding:try encoder.encode(mapping.4),as:UTF8.self),"--target-column",mapping.5,"--composite-source",provenance.path,"--composite-source-sha256",sha(sourceBytes)]
        }
        for index in 0...1 {
            let out=temp.appendingPathComponent("model-\(index)")
            let result=try worker(python,root:root,temp:temp,args:trainArguments(index:index,output:out))
            XCTAssertEqual(result.code,0,result.text);guard result.code==0 else{return};modelOutputs.append(out)
        }
        XCTAssertEqual(try Data(contentsOf:modelOutputs[0].appendingPathComponent("metrics.json")),try Data(contentsOf:modelOutputs[1].appendingPathComponent("metrics.json")))
        let script="""
        import importlib.util,json,sys,torch
        from pathlib import Path
        from runner.engine import GraphModel
        roots=[Path(p) for p in sys.argv[1:]]
        def equal(a,b):
            if isinstance(a,torch.Tensor): assert torch.equal(a,b)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a: equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b): equal(x,y)
            else: assert a==b,(a,b)
        states=[torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True) for p in roots]
        equal(states[0],states[1])
        manifests=[json.loads((p/'manifest.json').read_text()) for p in roots]
        equal(manifests[0]['provenance'],manifests[1]['provenance'])
        assert all(m['settings']['partitionMode']=='prepared' for m in manifests)
        assert all(m['settings']['splitSeed'] is None and m['settings']['sampleLimit'] is None for m in manifests)
        assert (roots[0]/'preparation/manifest.json').read_bytes()==(roots[1]/'preparation/manifest.json').read_bytes()
        assert (roots[0]/'composite/source.json').read_bytes()==(roots[1]/'composite/source.json').read_bytes()
        reference=None
        for index,p in enumerate(roots):
            graph=json.loads((p/'graph.json').read_text());model=GraphModel(graph);model.load_state_dict(states[index]['modelState'],strict=True)
            spec=importlib.util.spec_from_file_location('exported'+str(index),p/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            exported=module.Model();exported.load_state_dict(states[index]['modelState'],strict=True)
            assert set(exported.state_dict())==set(model.state_dict())
            with torch.no_grad():
                x=torch.tensor([[0.,0.],[1.,-1.],[-2.,3.],[.5,.25]])
                actual=model(x);torch.testing.assert_close(exported(x),actual,rtol=0,atol=0)
                if reference is not None: torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                reference=actual
        """
        let parity=try worker(python,root:root,temp:temp,args:["-c",script]+modelOutputs.map(\.path))
        XCTAssertEqual(parity.code,0,parity.text)
        // Preserve the original pin, then mutate bytes before dispatch: no successful training may appear.
        let refused=temp.appendingPathComponent("refused"),args=try trainArguments(index:1,output:refused)
        try Data("{}".utf8).write(to:preparationOutputs[1].appendingPathComponent("manifest.json"))
        let rejection=try worker(python,root:root,temp:temp,args:args)
        XCTAssertNotEqual(rejection.code,0,rejection.text)
        XCTAssertTrue(rejection.text.lowercased().contains("manifest"),rejection.text)
        XCTAssertFalse(FileManager.default.fileExists(atPath:refused.appendingPathComponent("manifest.json").path))
    }
}
