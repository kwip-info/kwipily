import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowConsolePresentationTests:XCTestCase {
    @MainActor func testSelectedComponentRequiresExactDomainAndLocalIDAndExcludesRunEvents() {
        let console=WorkflowRunConsole()
        let selected=WorkflowNodeAddress(domain:.data,localID:"shared")
        console.append(phase:.preparing,message:"source",component:selected)
        console.append(phase:.training,message:"model",component:.init(domain:.model,localID:"shared"))
        console.append(phase:.preparing,message:"another source",component:.init(domain:.data,localID:"other"))
        console.append(phase:.training,message:"global")
        let scoped=WorkflowConsolePresentation.visibleEntries(console.entries,component:selected)
        XCTAssertEqual(scoped.map(\.message),["source"])
        XCTAssertEqual(WorkflowConsolePresentation.visibleEntries(console.entries).count,4)
        XCTAssertEqual(console.entries.count,4,"Scoping presentation must not discard captured global events")
    }

    @MainActor func testScopeComposesWithExistingProblemSearchAndPausedSnapshotWithoutLeakage() {
        let console=WorkflowRunConsole()
        let selected=WorkflowNodeAddress(domain:.model,localID:"model-a")
        console.append(phase:.training,level:.warning,message:"Needs review",component:selected)
        console.append(phase:.training,level:.info,message:"review informational",component:selected)
        console.append(phase:.training,level:.error,message:"review foreign",component:.init(domain:.data,localID:"model-a"))
        let paused=console.entries
        console.append(phase:.training,level:.error,message:"review later",component:selected)
        let visible=WorkflowConsolePresentation.visibleEntries(paused,component:selected,problemsOnly:true,query:"REVIEW")
        XCTAssertEqual(visible.map(\.message),["Needs review"])
        XCTAssertTrue(WorkflowConsolePresentation.visibleEntries(paused,component:selected,query:"no-match").isEmpty)
    }
}
