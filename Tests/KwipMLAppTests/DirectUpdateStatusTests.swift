import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp
#if KWIPML_DIRECT_UPDATES
import Sparkle

@MainActor final class DirectUpdateStatusTests:XCTestCase {
    func testSuccessfulProgressReplacesSignInRefusalWithoutClaimingInstallation() {
        let store=DirectUpdateStore()
        store.transition(to:.signInRequired)
        XCTAssertFalse(store.message.contains("Sign in"))
        XCTAssertTrue(store.message.contains("refused"))
        for (outcome,text) in [
            (UpdateDiagnosticOutcome.checking,"Checking for a compatible v1 update…"),
            (.offerAvailable,"A verified v1 update is available. You choose when to download and install."),
            (.downloading,"Downloading the update…"),
            (.finished,"Update cycle finished. Check the app version after reopening to confirm installation.")
        ] {
            store.transition(to:outcome)
            XCTAssertEqual(store.diagnosticOutcome,outcome)
            XCTAssertEqual(store.message,text)
            XCTAssertFalse(store.message.contains("Sign in"))
        }
        store.transition(to:.activeWork,message:"Finish or cancel active work before downloading an update.")
        store.transition(to:.activeWork)
        XCTAssertEqual(store.message,"Finish or cancel active work before downloading an update.")
    }

    func testCompletedCallbacksReplaceEarlierRefusalWithCurrentSanitizedStatus() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".app")
        let contents=directory.appendingPathComponent("Contents")
        try FileManager.default.createDirectory(at:contents,withIntermediateDirectories:true)
        try PropertyListSerialization.data(fromPropertyList:[
            "CFBundleIdentifier":"info.kwip.tests.status."+UUID().uuidString,
            "CFBundleVersion":"1","SUFeedURL":"https://foreign.invalid/never-requested"
        ],format:.xml,options:0).write(to:contents.appendingPathComponent("Info.plist"))
        defer {try? FileManager.default.removeItem(at:directory)}
        let bundle=try XCTUnwrap(Bundle(url:directory))
        let updater=SPUUpdater(hostBundle:bundle,applicationBundle:bundle,
            userDriver:SPUStandardUserDriver(hostBundle:bundle,delegate:nil),delegate:nil)
        let store=DirectUpdateStore()
        XCTAssertThrowsError(try store.updater(updater,mayPerform:.updates))
        XCTAssertEqual(store.message,"The update destination could not be verified.")
        store.updater(updater,didAbortWithError:NSError(domain:"SUSparkleErrorDomain",code:1001))
        XCTAssertEqual(store.diagnosticOutcome,.noUpdate)
        XCTAssertEqual(store.message,"No newer compatible update is available.")
        store.updater(updater,didAbortWithError:NSError(domain:NSURLErrorDomain,code:NSURLErrorNotConnectedToInternet,
            userInfo:[NSLocalizedDescriptionKey:"PRIVATE token /Users/private/model"]))
        XCTAssertEqual(store.diagnosticOutcome,.networkFailed)
        XCTAssertEqual(store.message,"The update check or download could not reach the service. Try again when connected.")
        store.userDidCancelDownload(updater)
        XCTAssertEqual(store.diagnosticOutcome,.cancelled)
        XCTAssertEqual(store.message,"Update cancelled. You can check again when ready.")
        store.updater(updater,didFinishUpdateCycleFor:.updates,error:nil)
        XCTAssertEqual(store.diagnosticOutcome,.cancelled)
        XCTAssertEqual(store.message,"Update cancelled. You can check again when ready.")
        XCTAssertFalse(store.message.contains("PRIVATE"))
        // No updater start/check, credentials, real tokens or network calls.
    }
}
#endif
