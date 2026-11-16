import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DirectUpdatePolicyTests:XCTestCase {
    private let digest=String(repeating:"a",count:64)
    private func offer(version:String="1.0.1",build:String="8",url:String?=nil,size:UInt64=900) -> DirectUpdateOffer {
        DirectUpdateOffer(version:version,build:build,url:URL(string:url ?? DesktopDeployment.origin+"/downloads/8/\(digest)/KwipML.dmg"),size:size,signedFeed:true)
    }
    func testOnlyNewerSignedV1InstallerOnFixedHostIsEligible() {
        XCTAssertTrue(DirectUpdatePolicy.accepts(offer(),installedBuild:"7"))
        for invalid in [offer(version:"2.0.0"),offer(version:"1.0.1\n"),offer(build:"7"),offer(build:"08"),offer(size:0),offer(size:6*1024*1024*1024),
          offer(url:"http://apps.kwip.info/downloads/8/\(digest)/KwipML.dmg"),
          offer(url:"https://other.invalid/downloads/8/\(digest)/KwipML.dmg"),
          offer(url:DesktopDeployment.origin+"/downloads/8/\(digest)/KwipML.dmg?token=secret"),
          offer(url:DesktopDeployment.origin+"/downloads/8/\(digest)/KwipML.dmg#fragment"),
          offer(url:DesktopDeployment.origin+"/downloads/8/\(digest)/KwipML.dmg/"),
          offer(url:DesktopDeployment.origin+"/downloads/9/\(digest)/KwipML.dmg")] {
            XCTAssertFalse(DirectUpdatePolicy.accepts(invalid,installedBuild:"7"))
        }
        var unsigned=offer();unsigned.signedFeed=false
        XCTAssertFalse(DirectUpdatePolicy.accepts(unsigned,installedBuild:"7"))
        XCTAssertFalse(DirectUpdatePolicy.accepts(offer(),installedBuild:"unknown"))
    }
    func testHostCredentialsCannotBeEmbeddedInURLOrSentToOtherPortsOrEncodedPaths() {
        for url in ["https://user:pass@apps.kwip.info/downloads/8/\(digest)/KwipML.dmg",
          "https://apps.kwip.info:443/downloads/8/\(digest)/KwipML.dmg",
          DesktopDeployment.origin+"/downloads/8/\(digest)/%4bwipML.dmg",
          "http://user:pass@127.0.0.1:8768/downloads/8/\(digest)/KwipML.dmg",
          "http://127.0.0.1:8769/downloads/8/\(digest)/KwipML.dmg"] {
            XCTAssertFalse(DirectUpdatePolicy.accepts(offer(url:url),installedBuild:"7"))
        }
    }
    func testCheckAndInstallAdmissionProtectAllActivityAndRequireExplicitDownload() {
        XCTAssertTrue(DirectUpdatePolicy.canCheck(production:true,configured:true,activeWork:false))
        XCTAssertFalse(DirectUpdatePolicy.canCheck(production:true,configured:true,activeWork:true))
        XCTAssertFalse(DirectUpdatePolicy.canCheck(production:false,configured:true,activeWork:false))
        XCTAssertFalse(DirectUpdatePolicy.canCheck(production:true,configured:false,activeWork:false))
    }
    func testConfigurationRequiresSignedFeedWithoutTimeoutFallbackAndAllIsolatedServices() {
        let valid:[String:Any]=["CFBundleIdentifier":DesktopDeployment.bundleIdentifier,"KWIPMLDirectUpdatesEnabled":true,
          "SUFeedURL":DirectUpdatePolicy.feedURL,"SUPublicEDKey":DirectUpdatePolicy.publicKey,
          "SURequireSignedFeed":true,"SUVerifyUpdateBeforeExtraction":true,"SUSignedFeedFailureExpirationInterval":0,
          "SUAllowsAutomaticUpdates":false,"SUAutomaticallyUpdate":false,
          "SUEnableDownloaderService":true,"SUEnableInstallerLauncherService":true,
          "SUEnableInstallerConnectionService":true,"SUEnableInstallerStatusService":true]
        XCTAssertTrue(DirectUpdatePolicy.configured(valid))
        for key in valid.keys {
            var missing=valid;missing.removeValue(forKey:key)
            XCTAssertFalse(DirectUpdatePolicy.configured(missing),key)
        }
        for change:[String:Any] in [["SUVerifyUpdateBeforeExtraction":false],["SUSignedFeedFailureExpirationInterval":3600],["SURequireSignedFeed":false],
          ["SUAutomaticallyUpdate":true],["SUAllowsAutomaticUpdates":true],["SUFeedURL":"https://other.invalid/feed"],
          ["CFBundleIdentifier":DesktopDeployment.isStaging ? "info.kwip.KwipML" : "info.kwip.KwipML.POC"]] {
            XCTAssertFalse(DirectUpdatePolicy.configured(valid.merging(change){_,new in new}))
        }
    }

}

extension DirectUpdatePolicyTests {
    func testCompiledEditionKeepsUpdateTrustOnItsFixedOriginAndParent() {
        let staging = ProcessInfo.processInfo.environment["KWIPML_COMMERCE_STAGING"] == "1"
        let origin = staging ? "http://127.0.0.1:8768" : "https://apps.kwip.info"
        let other = staging ? "https://apps.kwip.info" : "http://127.0.0.1:8768"
        let bundle = staging ? "info.kwip.KwipML.POC" : "info.kwip.KwipML"
        XCTAssertEqual(DirectUpdatePolicy.feedURL, origin + "/updates/v1/arm64/appcast.xml")
        let digest = String(repeating: "a", count: 64)
        let offered = DirectUpdateOffer(version: "1.0.1", build: "2",
            url: URL(string: origin + "/downloads/2/\(digest)/KwipML.dmg"), size: 900, signedFeed: true)
        XCTAssertTrue(DirectUpdatePolicy.accepts(offered, installedBuild: "1"))
        for url in [other + "/downloads/2/\(digest)/KwipML.dmg",
                    "http://127.0.0.1:8769/downloads/2/\(digest)/KwipML.dmg",
                    origin + "/downloads/2/\(digest)/KwipML.dmg?token=private",
                    origin + "/downloads/2/\(digest)/KwipML.dmg#fragment"] {
            let invalid = DirectUpdateOffer(version: "1.0.1", build: "2", url: URL(string: url), size: 900, signedFeed: true)
            XCTAssertFalse(DirectUpdatePolicy.accepts(invalid, installedBuild: "1"))
        }
        var info: [String: Any] = ["CFBundleIdentifier": bundle, "KWIPMLDirectUpdatesEnabled": true,
            "SUFeedURL": origin + "/updates/v1/arm64/appcast.xml", "SUPublicEDKey": DirectUpdatePolicy.publicKey,
            "SURequireSignedFeed": true, "SUVerifyUpdateBeforeExtraction": true, "SUSignedFeedFailureExpirationInterval": 0,
            "SUAllowsAutomaticUpdates": false, "SUAutomaticallyUpdate": false,
            "SUEnableDownloaderService": true, "SUEnableInstallerLauncherService": true,
            "SUEnableInstallerConnectionService": true, "SUEnableInstallerStatusService": true]
        XCTAssertTrue(DirectUpdatePolicy.configured(info))
        info["CFBundleIdentifier"] = staging ? "info.kwip.KwipML" : "info.kwip.KwipML.POC"
        XCTAssertFalse(DirectUpdatePolicy.configured(info))
    }
}

extension DirectUpdatePolicyTests {
    func testEveryArchiveRequestDropsCapturedHeadersAndRequiresExactValidatedDestinationWithoutCredentials() {
        let url=URL(string:DesktopDeployment.origin+"/downloads/2/"+String(repeating:"a",count:64)+"/KwipML.dmg")!
        let valid=DirectUpdateOffer(version:"1.0.1",build:"2",url:url,size:900,signedFeed:true)
        var unsigned=valid;unsigned.signedFeed=false
        let foreign=URL(string:"https://foreign.invalid/collect")!
        let foreignSignedOffer=DirectUpdateOffer(version:"1.0.1",build:"2",url:foreign,size:900,signedFeed:true)
        for (offer,destination) in [(valid,URL(string:"https://foreign.invalid/collect")!),
                                    (unsigned,url),(foreignSignedOffer,foreign)] {
            let request=NSMutableURLRequest(url:destination)
            request.setValue("Bearer synthetic-captured",forHTTPHeaderField:"Authorization")
            let reads=0
            XCTAssertFalse(DirectUpdatePolicy.prepareDownloadRequest(request,offer:offer,installedBuild:"1"))
            XCTAssertEqual(reads,0)
            XCTAssertNil(request.value(forHTTPHeaderField:"Authorization"))
            XCTAssertEqual(request.url?.absoluteString,"about:blank")
        }
        let request=NSMutableURLRequest(url:url)
        request.setValue("Bearer synthetic-captured",forHTTPHeaderField:"Authorization")
        request.setValue("old-account-session",forHTTPHeaderField:"Cookie")
        XCTAssertTrue(DirectUpdatePolicy.prepareDownloadRequest(request,offer:valid,installedBuild:"1"))
        XCTAssertNil(request.value(forHTTPHeaderField:"Cookie"))
        XCTAssertFalse(request.httpShouldHandleCookies)
        XCTAssertNil(request.value(forHTTPHeaderField:"Authorization"))
        XCTAssertTrue(DirectUpdatePolicy.prepareDownloadRequest(request,offer:valid,installedBuild:"1"))
        XCTAssertNil(request.value(forHTTPHeaderField:"Authorization"))
        XCTAssertEqual(request.url,url)
    }
}
