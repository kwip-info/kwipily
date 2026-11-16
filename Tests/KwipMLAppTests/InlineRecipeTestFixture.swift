import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Every operation delegates to the live inline Train draft; no standalone editor is recreated.
@MainActor extension XCTestCase {
    func inlineRecipeFixture(_ id:String, role:String, config:[String:JSONValue]=[:]) throws -> (WorkflowAuthoringStore,WorkflowNodeEditingAuthority,TrainInlineRecipeDraft) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let address=WorkflowExamples.trainAddress, library=ProjectLibrary(directory:directory)
        let recipe:JSONValue
        switch role {
        case "optimizer":recipe=try OptimizerConfiguration(componentID:id,values:config).recipeValue
        case "scheduler":recipe=try SchedulerConfiguration(componentID:id,values:config).recipeValue
        case "objective":recipe=try LossConfiguration(componentID:id,values:config).recipeValue
        default:throw GraphConfigurationError("Unsupported scalar test role")
        }
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,payload:.train(config:[role:recipe]))])
        document.schemaVersion=13
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let authority=try WorkflowNodeEditingAuthority(owner:owner,address:address)
        return(owner,authority,TrainInlineRecipeDraft(session:.init(authority:authority),role:role))
    }
}

@MainActor extension TrainInlineRecipeDraft {
    var draftConfiguration:[String:JSONValue] {configuration}
    var committedConfiguration:[String:JSONValue] {
        guard case .object(let envelope)=session.authority.configuration[role],case .object(let config)=envelope["config"] else {return [:]}
        return config
    }
    var expectedConfiguration:[String:JSONValue] {session.expected}
    func numericOrigin(key:String)->String {numericOrigin(key)}
    func currentNumericOrigin(key:String)->String? {currentNumericOrigin(key)}
    func setNumericPending(origin:String,key:String,pending:Bool) {setNumericPending(key,origin:origin,pending:pending)}
    func discard()->Bool {revert()}
}
