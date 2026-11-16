import XCTest
import KwipMLCore
@testable import KwipMLApp

final class Data02DestinationContractTests:XCTestCase {
    func testDestinationSidecarMustBeCapturedAtExactOwnedPath()throws {
        let root=URL(fileURLWithPath:"/tmp/workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
        let hash=String(repeating:"a",count:64)
        let names=["input/graph.json","input/bindings.json","input/node-labels.json","input/destinations.json"]
        let files=names.map {WorkflowArchiveFile(path:$0,bytes:2,sha256:hash)}
        let args=["--execute-dataflow",root.appendingPathComponent(names[0]).path,"--source-bindings",root.appendingPathComponent(names[1]).path,"--dataflow-node-labels",root.appendingPathComponent(names[2]).path,"--destination-bindings",root.appendingPathComponent(names[3]).path,"--output",root.appendingPathComponent("artifacts").path,"--dataflow-cache-policy","off"]
        XCTAssertNoThrow(try WorkflowPreparationToolContract.validate(arguments:args,files:files,directory:root))
        var wrong=args;wrong[7]="/tmp/destinations.json"
        XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments:wrong,files:files,directory:root))
        XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments:args,files:Array(files.dropLast()),directory:root))
    }
}
