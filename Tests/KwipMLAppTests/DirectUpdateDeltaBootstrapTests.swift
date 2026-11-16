import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp
#if KWIPML_DIRECT_UPDATES
import Sparkle
#endif

final class DirectUpdateDeltaBootstrapTests:XCTestCase {
    private let digest=String(repeating:"a",count:64)
    private let signature=Data(repeating:7,count:64).base64EncodedString()
    private func delta(base:String="14",target:String="15",url:String?=nil,size:UInt64=100)->DirectUpdateOffer {
        DirectUpdateOffer(version:"1.2.1",build:target,url:URL(string:url ?? DesktopDeployment.origin+"/downloads/\(target)/\(digest)/KwipML-from-\(base).delta"),size:size,signedFeed:true,isDelta:true,deltaFrom:base,artifactSignature:signature)
    }
    private func full(target:String="15")->DirectUpdateOffer {
        DirectUpdateOffer(version:"1.2.1",build:target,url:URL(string:DesktopDeployment.origin+"/downloads/\(target)/\(digest)/KwipML.dmg"),size:900,signedFeed:true)
    }
    func testExactBaseSignedDeltaAndUnchangedFullAreAdmitted() {
        XCTAssertTrue(DirectUpdatePolicy.accepts(delta(),installedBuild:"14"))
        XCTAssertTrue(DirectUpdatePolicy.accepts(full(),installedBuild:"14"))
        XCTAssertFalse(DirectUpdatePolicy.accepts(delta(),installedBuild:"13"))
        XCTAssertFalse(DirectUpdatePolicy.accepts(delta(base:"014"),installedBuild:"14"))
        XCTAssertFalse(DirectUpdatePolicy.accepts(delta(target:"14"),installedBuild:"14"))
        XCTAssertFalse(DirectUpdatePolicy.accepts(delta(size:0),installedBuild:"14"))
        XCTAssertFalse(DirectUpdatePolicy.accepts(delta(size:5*1024*1024*1024+1),installedBuild:"14"))
        var malformed=delta();malformed.deltaFrom=nil
        XCTAssertFalse(DirectUpdatePolicy.accepts(malformed,installedBuild:"14"))
        malformed=delta();malformed.isDelta=false
        XCTAssertFalse(DirectUpdatePolicy.accepts(malformed,installedBuild:"14"))
        malformed=delta();malformed.artifactSignature="unsigned"
        XCTAssertFalse(DirectUpdatePolicy.accepts(malformed,installedBuild:"14"))
        malformed=delta();malformed.signedFeed=false
        XCTAssertFalse(DirectUpdatePolicy.accepts(malformed,installedBuild:"14"))
    }
    func testDeltaURLAuthorityCannotEscapeOrContradictMetadata() {
        let origin=DesktopDeployment.origin
        let path="/downloads/15/\(digest)/KwipML-from-14.delta"
        for url in ["https://foreign.invalid"+path,"http://apps.kwip.info"+path,"https://apps.kwip.info:443"+path,
                    "https://user:pass@apps.kwip.info"+path,origin+path+"?token=private",origin+path+"#fragment",origin+path+"/",
                    origin+path.replacingOccurrences(of:"from-14",with:"from-13"),origin+path.replacingOccurrences(of:"/15/",with:"/16/"),
                    origin+path.replacingOccurrences(of:"KwipML",with:"%4bwipML"),origin+path.replacingOccurrences(of:digest,with:digest.uppercased()),
                    origin+path.replacingOccurrences(of:digest,with:String(digest.dropLast())),origin+"/downloads/15/../"+path,
                    origin+path.replacingOccurrences(of:".delta",with:".dmg")] {
            XCTAssertFalse(DirectUpdatePolicy.accepts(delta(url:url),installedBuild:"14"),url)
        }
    }
    func testEveryConcreteFallbackIsTargetBoundAndRemovesCredentialsOrRefuses()throws {
        let offer=delta();let target=try XCTUnwrap(DirectUpdateTarget(offer:offer,installedBuild:"14"))
        let first=NSMutableURLRequest(url:try XCTUnwrap(offer.url));first.setValue("Bearer captured",forHTTPHeaderField:"Authorization")
        XCTAssertTrue(DirectUpdatePolicy.prepareDownloadRequest(first,offer:offer,installedBuild:"14",target:target,activeWork:false))
        let fallback=full();let request=NSMutableURLRequest(url:try XCTUnwrap(fallback.url));request.setValue("Bearer captured",forHTTPHeaderField:"Authorization")
        XCTAssertTrue(DirectUpdatePolicy.prepareDownloadRequest(request,offer:fallback,installedBuild:"14",target:target,activeWork:false))
        XCTAssertNil(request.value(forHTTPHeaderField:"Authorization"))
        XCTAssertTrue(DirectUpdatePolicy.prepareDownloadRequest(request,offer:fallback,installedBuild:"14",target:target,activeWork:false))
        XCTAssertNil(request.value(forHTTPHeaderField:"Authorization"));XCTAssertEqual(request.url,fallback.url)
        for (offered,active) in [(full(target:"16"),false),(fallback,true)] {
            let denied=NSMutableURLRequest(url:try XCTUnwrap(offered.url));denied.setValue("Bearer captured",forHTTPHeaderField:"Authorization");let reads=0
            XCTAssertFalse(DirectUpdatePolicy.prepareDownloadRequest(denied,offer:offered,installedBuild:"14",target:target,activeWork:active))
            XCTAssertEqual(reads,0);XCTAssertNil(denied.value(forHTTPHeaderField:"Authorization"));XCTAssertEqual(denied.url?.absoluteString,"about:blank")
        }
    }
    func testCycleBindingStickyMetadataAndExplicitResets()throws {
        var cycle=DirectUpdateCycleAdmission()
        XCTAssertNil(cycle.target)
        XCTAssertFalse(cycle.admit(delta(),installedBuild:"14",activeWork:true))
        XCTAssertNil(cycle.target)
        XCTAssertTrue(cycle.admit(delta(),installedBuild:"14",activeWork:false))
        let target=try XCTUnwrap(cycle.target)
        XCTAssertTrue(cycle.admit(delta(),installedBuild:"14",activeWork:false),"Repeated same-target callback remains valid")
        XCTAssertTrue(cycle.admit(full(),installedBuild:"14",activeWork:false),"Full fallback retains the admitted target")
        XCTAssertFalse(cycle.admit(full(target:"16"),installedBuild:"14",activeWork:false))
        XCTAssertEqual(cycle.target,target,"Another offer cannot replace the cycle's target")
        cycle.observe([["sparkle:deltas":[["sparkle:deltaFrom":"14"],["sparkle:deltaFrom":"14"]]]])
        XCTAssertFalse(cycle.metadataValid);XCTAssertNil(cycle.target)
        cycle.observe([[:]])
        XCTAssertFalse(cycle.metadataValid,"A later valid observation cannot repair an invalid cycle")
        XCTAssertFalse(cycle.admit(full(),installedBuild:"14",activeWork:false))
        cycle.reset();XCTAssertTrue(cycle.metadataValid);XCTAssertNil(cycle.target)
        XCTAssertTrue(cycle.admit(full(),installedBuild:"14",activeWork:false))
        cycle.reset();XCTAssertNil(cycle.target)
    }
    func testMalformedDuplicateOrWrongTypeRawDeltaMetadataIsRefused() {
        let good:[String:Any]=["sparkle:deltaFrom":"14","url":delta().url!.absoluteString,"length":"100","sparkle:edSignature":signature]
        XCTAssertTrue(DirectUpdatePolicy.deltaMetadataValid([:]))
        XCTAssertTrue(DirectUpdatePolicy.deltaMetadataValid(["sparkle:deltas":[good]]))
        for raw:Any in ["not an array",[good,good],[["sparkle:deltaFrom":14]],[["sparkle:deltaFrom":["14","14"]]],[["sparkle:deltaFrom":"014"]],[["sparkle:deltaFrom":"14\n"]],Array(repeating:good,count:17)] {
            XCTAssertFalse(DirectUpdatePolicy.deltaMetadataValid(["sparkle:deltas":raw]))
        }
    }
#if KWIPML_DIRECT_UPDATES
    @MainActor func testPinnedSDKDelegateSelectorsActuallyReachAdmissionAndCycleCallbacks() {
        let store=DirectUpdateStore()
        for selector in ["updater:didFinishLoadingAppcast:","updater:shouldProceedWithUpdate:updateCheck:error:",
                         "updater:willDownloadUpdate:withRequest:","updater:mayPerformUpdateCheck:error:",
                         "userDidCancelDownload:","updater:didAbortWithError:",
                         "updater:didFinishUpdateCycleForUpdateCheck:error:"] {
            XCTAssertTrue(store.responds(to:NSSelectorFromString(selector)),selector)
        }
    }
    func testPinnedSDKParentRetainsRawDeltaArrayWhileItsDerivedMapCollapsesDuplicateBases()throws {
        let deltaEnclosure:[String:Any]=["url":delta().url!.absoluteString,"length":"100","sparkle:deltaFrom":"14","sparkle:edSignature":signature]
        let fullEnclosure:[String:Any]=["url":full().url!.absoluteString,"length":"900","sparkle:edSignature":signature]
        let properties:[String:Any]=["sparkle:version":"15","sparkle:shortVersionString":"1.2.1","enclosure":fullEnclosure,"sparkle:deltas":[deltaEnclosure]]
        let parent=try XCTUnwrap(SUAppcastItem(dictionary:properties))
        XCTAssertFalse(parent.isDeltaUpdate)
        XCTAssertTrue(DirectUpdatePolicy.deltaMetadataValid(parent.propertiesDictionary))
        XCTAssertEqual((parent.propertiesDictionary["sparkle:deltas"] as? [[String:Any]])?.count,1)
        let selected=try XCTUnwrap(parent.deltaUpdates?["14"])
        XCTAssertTrue(selected.isDeltaUpdate)
        let mapped=try XCTUnwrap(DirectUpdateSparkleAdapter.offer(selected))
        XCTAssertEqual(mapped.deltaFrom,"14");XCTAssertEqual(mapped.url,delta().url)
        XCTAssertEqual(mapped.artifactSignature,signature)
        XCTAssertNil(selected.propertiesDictionary["sparkle:deltas"],"SDK removes parent delta array from the derived selected item")
        XCTAssertFalse(mapped.signedFeed,"A synthetic public initializer has skipped signing authority")
        var second=deltaEnclosure;second["url"] = delta().url!.absoluteString.replacingOccurrences(of:digest,with:String(repeating:"b",count:64))
        var duplicateProperties=properties;duplicateProperties["sparkle:deltas"] = [deltaEnclosure,second]
        let duplicate=try XCTUnwrap(SUAppcastItem(dictionary:duplicateProperties))
        XCTAssertEqual(duplicate.deltaUpdates?.count,1,"Fixture exercises SDK's duplicate-base map collapse")
        XCTAssertEqual((duplicate.propertiesDictionary["sparkle:deltas"] as? [[String:Any]])?.count,2)
        XCTAssertFalse(DirectUpdatePolicy.deltaMetadataValid(duplicate.propertiesDictionary))
    }
    func testActualPinnedSDKAdapterReadsSelectedDeltaMetadataAndRejectsWrongTypes()throws {
        let enclosure:[String:Any]=["url":delta().url!.absoluteString,"length":"100","sparkle:deltaFrom":"14","sparkle:edSignature":signature]
        let selected=try XCTUnwrap(SUAppcastItem(dictionary:["sparkle:version":"15","sparkle:shortVersionString":"1.2.1","enclosure":enclosure]))
        XCTAssertTrue(selected.isDeltaUpdate)
        let mapped=try XCTUnwrap(DirectUpdateSparkleAdapter.offer(selected))
        XCTAssertTrue(mapped.isDelta);XCTAssertEqual(mapped.deltaFrom,"14");XCTAssertEqual(mapped.artifactSignature,signature)
        XCTAssertFalse(mapped.signedFeed,"Public synthetic initializer must never invent signed-feed authority")
        var bad=enclosure;bad["sparkle:deltaFrom"] = ["14","14"]
        let invalid=try XCTUnwrap(SUAppcastItem(dictionary:["sparkle:version":"15","sparkle:shortVersionString":"1.2.1","enclosure":bad]))
        XCTAssertNil(DirectUpdateSparkleAdapter.offer(invalid))
    }
#endif
}
