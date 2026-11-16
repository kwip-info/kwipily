import Foundation
import XCTest
@testable import KwipMLApp
#if KWIPML_DIRECT_UPDATES
import Sparkle

@MainActor final class DirectUpdateCredentialBoundaryTests:XCTestCase {
    func testStoredForeignFeedCannotReplaceCompiledOriginBeforeAnyNetworkOrCredentialRead() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".app")
        let contents=directory.appendingPathComponent("Contents")
        try FileManager.default.createDirectory(at:contents,withIntermediateDirectories:true)
        let identifier="info.kwip.tests.updates."+UUID().uuidString
        let info:[String:Any]=["CFBundleIdentifier":identifier,"CFBundleName":"Synthetic updater",
                              "CFBundleVersion":"1","SUFeedURL":DirectUpdatePolicy.feedURL]
        try PropertyListSerialization.data(fromPropertyList:info,format:.xml,options:0)
            .write(to:contents.appendingPathComponent("Info.plist"))
        defer {
            UserDefaults.standard.removePersistentDomain(forName:identifier)
            try? FileManager.default.removeItem(at:directory)
        }
        let bundle=try XCTUnwrap(Bundle(url:directory))
        let defaults=try XCTUnwrap(UserDefaults(suiteName:identifier))
        defaults.set("https://foreign.invalid/collect",forKey:"SUFeedURL")
        let driver=SPUStandardUserDriver(hostBundle:bundle,delegate:nil)
        let unbound=SPUUpdater(hostBundle:bundle,applicationBundle:bundle,userDriver:driver,delegate:nil)
        XCTAssertEqual(unbound.feedURL?.absoluteString,"https://foreign.invalid/collect","Fixture must exercise Sparkle's actual stored-feed precedence")
        let store=DirectUpdateStore()
        let protected=SPUUpdater(hostBundle:bundle,applicationBundle:bundle,userDriver:driver,delegate:store)
        XCTAssertEqual(protected.feedURL?.absoluteString,DirectUpdatePolicy.feedURL)
        XCTAssertTrue(store.responds(to:NSSelectorFromString("feedURLStringForUpdater:")))
        unbound.httpHeaders=["Authorization":"Bearer synthetic-captured"]
        XCTAssertThrowsError(try store.updater(unbound,mayPerform:.updates)) { error in
            XCTAssertEqual((error as NSError).localizedDescription,"The update destination could not be verified.")
        }
        XCTAssertEqual(store.diagnosticOutcome,.configurationFailed)
        XCTAssertEqual(store.message,"The update destination could not be verified.")
        XCTAssertNil(unbound.httpHeaders)
        // Origin refusal precedes Keychain access. No updater start/check or requests.
    }
}
#endif
