import XCTest
@testable import KwipMLCore

final class Data02LineageTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func address(_ id:String)->WorkflowNodeAddress {.init(domain:.data,localID:id)}
    func testSixCurrentModesPreservePartitionSourceAndFittedStateWithoutInventedRoles()throws {
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        for (name,id,mode) in [("expression filter","kwip.data.filterRows","expression"),("arithmetic formula","kwip.data.formula",""),("pattern filter","kwip.data.filterRows","pattern"),("extract formula","kwip.data.formula",""),("replace formula","kwip.data.formula",""),("Table Output","kwip.data.outputDelimited","")] {
            let operation=try XCTUnwrap(registry.operatorDefinition(id:id))
            var configuration=operation.defaultConfig
            if id == "kwip.data.filterRows" {configuration["mode"] = .string(mode)}
            if id == "kwip.data.formula" {
                let expression = name == "arithmetic formula" ? "[feature] + 1" : name == "extract formula" ? #"regex_extract([text], "([A-Z]+)", "search", "null", "", "1")"# : #"regex_replace([text], " +", " ", "all", "")"#
                configuration["operations"] = .array([.object(["id":.string("lineage-operation"),"expression":.string(expression),"nullPolicy":.string("preserve"),"outputs":.array([.object(["name":.string("derived"),"dtype":.string(name == "arithmetic formula" ? "float64":"string"),"write":.string("new")])])])])
            }
            let nodes:[WorkflowNode]=[
                .init(address:address("source"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
                .init(address:address("split"),payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":.integer(42),"validationPercent":.integer(25),"testPercent":.integer(25),"stratifyColumn":.string("label")])),
                .init(address:address("fit"),payload:.dataOperator(operatorID:"kwip.data.preprocess",config:CurrentPracticeConfiguration.scaledPartitions(["feature"]))),
                .init(address:address("tool"),payload:.dataOperator(operatorID:id,config:configuration))]
            let links=[("source","table","split","table",WorkflowEdgeKind.table),("split","training","fit","training",.table),("split","validation","fit","validation",.table),("split","test","fit","test",.table),("fit","validation","tool","table",.table)]
            let edges=links.enumerated().map {index,link in WorkflowEdge(id:"e\(index)",from:.init(address:address(link.0),port:link.1),to:.init(address:address(link.2),port:link.3),kind:link.4)}
            let document=WorkflowDocument(projectID:UUID().uuidString,name:"Lineage",nodes:nodes,edges:edges)
            let region=try WorkflowCompiler.dependencies(document,terminal:address("tool"),modelRegistry:.builtIn(),dataRegistry:registry)
            let evidence=try WorkflowDataLineage.resolve(region,dataRegistry:registry)
            for port in operation.outputs {
                guard case .rows(let rows)=evidence[.init(address:address("tool"),port:port.id)] else {return XCTFail("Missing reviewed output \(name).\(port.id)")}
                XCTAssertEqual(rows.source,address("source"),name)
                XCTAssertEqual(rows.split,address("split"),name)
                XCTAssertEqual(rows.role,"validation",name)
                XCTAssertEqual(rows.states,[address("fit")],name)
            }
            if id=="kwip.data.filterRows" {XCTAssertEqual(Set(operation.outputs.map(\.id)),["matching","nonmatching"])}
            else {XCTAssertEqual(operation.outputs.map(\.id),["table"])}
        }
    }
}
