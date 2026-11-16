import XCTest
@testable import KwipMLCore

final class BagTensorPortPresentationTests: XCTestCase {
    private func graph(packed:Bool=false,weighted:Bool=false,terminal:Bool=false)->GraphDocument {
        var source=BagSourceConfiguration.defaults
        source["layout"] = .string(packed ? "packed":"fixed")
        source["maximum_items_per_bag"] = 3
        source["per_sample_weights"] = .bool(weighted)
        source["include_last_offset"] = .bool(terminal)
        var table=EmbeddingBagConfiguration.defaults
        table["embedding_dim"] = 2;table["num_embeddings"] = 8
        table["mode"] = .string(weighted ? "sum":"mean")
        for key in ["layout","per_sample_weights","include_last_offset"] {table[key]=source[key]}
        let ports=["indices"]+(packed ? ["offsets"]:[])+(weighted ? ["per_sample_weights"]:[])
        return .init(schemaVersion:2,nodes:[.init(id:"bags",type:"bag_input",config:source),.init(id:"lookup",type:"embedding_bag",config:table),.init(id:"output",type:"output")],edges:ports.map {.init(id:$0,source:"bags",sourcePort:$0,target:"lookup",targetPort:$0)}+[.init(id:"reduced",source:"lookup",target:"output")])
    }
    private func check(_ descriptor:CanvasPortDescriptor,shape:String,dtype:String,role:String,file:StaticString=#filePath,line:UInt=#line) {
        XCTAssertEqual(descriptor.kind,.tensor,file:file,line:line)
        XCTAssertTrue(descriptor.declared.contains(shape),descriptor.declared.joined(separator:"; "),file:file,line:line)
        XCTAssertTrue(descriptor.declared.contains(dtype),file:file,line:line)
        XCTAssertEqual(descriptor.role,role,file:file,line:line)
        XCTAssertTrue(descriptor.measured.isEmpty,file:file,line:line)
        XCTAssertFalse(descriptor.declared.contains("Shape unresolved"),file:file,line:line)
    }
    func testFixedCoordinateSourceAndInputHavePhysicalCapacityAndActualSourceIdentity()throws {
        let registry=try ComponentRegistry.builtIn(),graph=graph(weighted:true)
        let result=GraphValidator.validate(graph,registry:registry,context:.preparedWorkflow)
        XCTAssertTrue(result.isValid,result.issues.map(\.message).joined(separator:"; "))
        for (port,dtype,role) in [("indices","int64","Category IDs"),("per_sample_weights","float32","Item influence weights")] {
            for (node,direction) in [("bags",PortDirection.output),("lookup",.input)] {
                check(result.tensorPortDescriptor(nodeID:node,port:port,direction:direction,graph:graph,registry:registry),shape:"[bags, 3]",dtype:dtype,role:role)
            }
            XCTAssertEqual(result.outputBagContracts[.init("bags",port)]?.sourceNodeID,"bags")
            XCTAssertNil(result.outputTensorContracts[.init("bags",port)])
        }
        let reduced=result.tensorPortDescriptor(nodeID:"lookup",port:"out",direction:.output,graph:graph,registry:registry)
        XCTAssertEqual(reduced.declared,["[N, 2]","float32"])
    }
    func testPackedCoordinatesHaveNoFakeObservationAxisAndBothOffsetConventions()throws {
        let registry=try ComponentRegistry.builtIn()
        for terminal in [false,true] {
            let graph=graph(packed:true,weighted:true,terminal:terminal)
            let result=GraphValidator.validate(graph,registry:registry,context:.preparedWorkflow)
            XCTAssertTrue(result.isValid)
            for (port,shape,dtype,role) in [("indices","[items]","int64","Category IDs"),("offsets",terminal ? "[bags + 1]":"[bags]","int64","Group boundaries"),("per_sample_weights","[items]","float32","Item influence weights")] {
                for (node,direction) in [("bags",PortDirection.output),("lookup",.input)] {
                    let descriptor=result.tensorPortDescriptor(nodeID:node,port:port,direction:direction,graph:graph,registry:registry)
                    check(descriptor,shape:shape,dtype:dtype,role:role)
                    XCTAssertFalse(descriptor.declared.contains(where:{$0.contains("[N,")}))
                }
            }
        }
    }
    func testDisabledCoordinateInputsAndOutputsStayUnknown()throws {
        let registry=try ComponentRegistry.builtIn(),graph=graph()
        let result=GraphValidator.validate(graph,registry:registry,context:.preparedWorkflow)
        for port in ["offsets","per_sample_weights"] {
            XCTAssertEqual(result.tensorPortDescriptor(nodeID:"bags",port:port,direction:.output,graph:graph,registry:registry).kind,.unknown)
            XCTAssertEqual(result.tensorPortDescriptor(nodeID:"lookup",port:port,direction:.input,graph:graph,registry:registry).kind,.unknown)
        }
    }
    func testPublishedWorkflowProjectionCarriesBagGeometryInsteadOfUnresolvedDenseShape()throws {
        let registry=try ComponentRegistry.builtIn()
        let workflow=try WorkflowExamples.preparedEmbeddingBagClassifier(device:"cpu")
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let plan=try WorkflowCompiler.compile(workflow,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:11)
        let projection=ModelTensorProjection(snapshot:plan.model,registry:registry)
        check(projection.port(instancePath:[],nodeID:"input",port:"indices",direction:.output),shape:"[bags, 3]",dtype:"int64",role:"Category IDs")
        check(projection.port(instancePath:[],nodeID:"lookup",port:"indices",direction:.input),shape:"[bags, 3]",dtype:"int64",role:"Category IDs")
    }
}
