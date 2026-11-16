import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class OptimizerRecipeInspectorTests:XCTestCase {
    private func fixture(_ id:String,config:[String:JSONValue]=[:])throws->(WorkflowAuthoringStore,WorkflowNodeEditingAuthority,TrainInlineRecipeDraft) {
        try inlineRecipeFixture(id,role:"optimizer",config:config)
    }
    func testAllFiveDraftsApplyCanonicalFieldsWithOwnAuthorityAndDiscardWithoutSourceMutation()throws {
        for id in OptimizerConfiguration.componentIDs.sorted() {
            let(owner,authority,context)=try fixture(id)
            XCTAssertEqual(context.componentID,id)
            XCTAssertEqual(context.fields,BundledOptimizerComponents.definition(id).fields)
            XCTAssertEqual(context.draftConfiguration,try OptimizerConfiguration(componentID:id,values:[:]).values)
            let initial=owner.document,origin=context.numericOrigin(key:"lr")
            XCTAssertTrue(context.setField("lr",value:.number(0.02),origin:context.numericOrigin(key:"lr")))
            XCTAssertTrue(context.hasChanges);XCTAssertTrue(owner.hasPendingInspectorEdits)
            XCTAssertEqual(owner.document,initial,"Draft must not alter source before Apply.")
            XCTAssertTrue(context.discard());XCTAssertEqual(owner.document,initial);XCTAssertFalse(owner.hasPendingInspectorEdits)
            XCTAssertTrue(context.setField("lr",value:.number(0.03),origin:context.numericOrigin(key:"lr")))
            XCTAssertTrue(context.apply());XCTAssertTrue(authority.isCurrent)
            XCTAssertEqual(context.committedConfiguration["lr"],.number(0.03));XCTAssertFalse(context.hasChanges)
            XCTAssertEqual(context.expectedConfiguration,authority.configuration)
            XCTAssertFalse(owner.hasPendingInspectorEdits)
            XCTAssertTrue(context.setField("lr",value:.number(0.04),origin:context.numericOrigin(key:"lr")));XCTAssertTrue(context.apply())
            XCTAssertEqual(context.committedConfiguration["lr"],.number(0.04),"Own first commit must renew authority for the next edit.")
        }
    }
    func testInterdependentDraftCanBeRepairedAndInactiveSettingsAreNeverErased()throws {
        let(owner,authority,context)=try fixture("optimizer_sgd",config:["momentum":.number(0.8),"dampening":.number(0.2)])
        XCTAssertTrue(context.setField("momentum",value:0,origin:context.numericOrigin(key:"momentum")))
        XCTAssertEqual(context.draftConfiguration["dampening"],.number(0.2))
        XCTAssertTrue(context.apply());XCTAssertEqual(context.committedConfiguration["dampening"],.number(0.2))
        XCTAssertTrue(context.setField("nesterov",value:true,origin:context.numericOrigin(key:"nesterov")))
        let before=owner.document
        XCTAssertNotNil(context.validationMessage);XCTAssertFalse(context.apply());XCTAssertEqual(owner.document,before)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(context.setField("momentum",value:.number(0.9),origin:context.numericOrigin(key:"momentum")))
        XCTAssertTrue(context.setField("dampening",value:0,origin:context.numericOrigin(key:"dampening")))
        XCTAssertNil(context.validationMessage);XCTAssertTrue(context.apply())
        XCTAssertEqual(context.committedConfiguration["nesterov"],true)
        XCTAssertFalse(context.setField("lr",value:true,origin:context.numericOrigin(key:"lr")))
        XCTAssertFalse(context.setField("arbitrary",value:0,origin:context.numericOrigin(key:"arbitrary")))
    }
    func testBusyRetirementAndForeignFieldTokensRefuseLateDraftsAndCommits()throws {
        let(owner,authority,context)=try fixture("optimizer_adam")
        let origin=context.numericOrigin(key:"lr")
        XCTAssertFalse(context.setField("lr",value:.number(0.2),origin:"foreign"))
        context.setNumericPending(origin:origin,key:"lr",pending:true)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        owner.setExecutionBusy(true)
        XCTAssertNil(context.currentNumericOrigin(key:"lr"));XCTAssertFalse(context.setField("lr",value:.number(0.2),origin:context.numericOrigin(key:"lr")))
        XCTAssertFalse(context.apply());owner.setExecutionBusy(false)
        context.setNumericPending(origin:origin,key:"lr",pending:false)
        XCTAssertTrue(context.setField("lr",value:.number(0.2),origin:context.numericOrigin(key:"lr")))
        authority.retire();let before=owner.document
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertFalse(context.apply());XCTAssertFalse(context.discard())
        XCTAssertFalse(context.setField("lr",value:.number(0.3),origin:context.numericOrigin(key:"lr")));XCTAssertEqual(owner.document,before)
    }
}
