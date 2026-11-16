import XCTest
import KwipMLCore
@testable import KwipMLApp
final class TypedDecisionAssetTests:XCTestCase {
 func testNestedExportIsExclusiveAndRetainsPins()throws {
  let root=try canonicalBackendTemporaryDirectory(),source=root.appendingPathComponent("source"),destination=root.appendingPathComponent("saved")
  try FileManager.default.createDirectory(at:source.appendingPathComponent("encoder"),withIntermediateDirectories:true)
  let bytes=Data("{}".utf8);try bytes.write(to:source.appendingPathComponent("encoder/config.json"))
  let pin=WorkflowArchiveFile(path:"encoder/config.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
  let pending=try WorkflowPendingExportPublication(destination:destination,directory:true);defer{pending.close()}
  try pending.writeModelFile(pin,from:source);XCTAssertThrowsError(try pending.writeModelFile(pin,from:source));try pending.commit()
  XCTAssertEqual(try Data(contentsOf:destination.appendingPathComponent(pin.path)),bytes)
  XCTAssertThrowsError(try WorkflowPendingExportPublication(destination:destination,directory:true))
  XCTAssertThrowsError(try ModelAssetToolContract.limit("source/encoder/code.py"))
 }
 func testHubNestedInventoryStaysNestedAndRequiresWholeProfile()throws {
  let names=["rl_agent_config.json","encoder/config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","model.safetensors"]
  let raw:[String:Any] = ["sha":String(repeating:"a",count:40),"siblings":names.map{["rfilename":"typed-decisions/"+$0,"size":100]}]
  let config=LoadModelConfiguration(source:.huggingFace,repository:"convaiinnovations/laya",subfolder:"typed-decisions")
  let snapshot=try ModelHubSnapshot.decode(JSONSerialization.data(withJSONObject:raw),configuration:config)
  XCTAssertEqual(Set(snapshot.files.map(\.path)),Set(names.map{"typed-decisions/"+$0}))
  var incomplete=raw;incomplete["siblings"]=names.dropLast().map{["rfilename":"typed-decisions/"+$0,"size":100]}
  XCTAssertThrowsError(try ModelHubSnapshot.decode(JSONSerialization.data(withJSONObject:incomplete),configuration:config))
 }
}
