import XCTest
@testable import KwipMLApp

final class FlatChoiceNavigationTests:XCTestCase {
    private let options:[FlatChoiceOption]=[
        .init(id:"blocked-start",title:"Unavailable",isEnabled:false),
        .init(id:"rose",title:"Rose"),
        .init(id:"blocked-middle",title:"Unavailable",isEnabled:false),
        .init(id:"teal",title:"Teal"),
        .init(id:"blocked-end",title:"Unavailable",isEnabled:false)
    ]
    func testOpeningUsesSelectedEnabledOptionOrFirstAvailable() {
        XCTAssertEqual(FlatChoiceNavigation.initialID(options:options,selection:"teal"),"teal")
        for selection in ["","missing","blocked-middle"] {
            XCTAssertEqual(FlatChoiceNavigation.initialID(options:options,selection:selection),"rose")
        }
    }
    func testArrowsSkipDisabledOptionsAndWrapInBothDirections() {
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:"rose",direction:.next),"teal")
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:"teal",direction:.next),"rose")
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:"teal",direction:.previous),"rose")
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:"rose",direction:.previous),"teal")
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:nil,direction:.next),"rose")
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:"removed",direction:.previous),"teal")
    }
    func testHomeEndAndSingleOptionDoNotLandOnUnavailableRows() {
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:"teal",direction:.first),"rose")
        XCTAssertEqual(FlatChoiceNavigation.move(options:options,from:"rose",direction:.last),"teal")
        let single=[FlatChoiceOption(id:"only",title:"Only")]
        for direction in FlatChoiceNavigation.Direction.allCases {
            XCTAssertEqual(FlatChoiceNavigation.move(options:single,from:"only",direction:direction),"only")
        }
    }
    func testEmptyOrDisabledListsNeverCommitAndDisabledControlRejectsStaleHighlight() {
        for list in [[],[FlatChoiceOption(id:"disabled",title:"Disabled",isEnabled:false)]] {
            XCTAssertNil(FlatChoiceNavigation.initialID(options:list,selection:"disabled"))
            for direction in FlatChoiceNavigation.Direction.allCases {
                XCTAssertNil(FlatChoiceNavigation.move(options:list,from:"disabled",direction:direction))
            }
            XCTAssertNil(FlatChoiceNavigation.commitID(options:list,highlighted:"disabled",controlEnabled:true))
        }
        XCTAssertNil(FlatChoiceNavigation.commitID(options:options,highlighted:"rose",controlEnabled:false))
        XCTAssertNil(FlatChoiceNavigation.commitID(options:options,highlighted:"blocked-middle",controlEnabled:true))
        XCTAssertNil(FlatChoiceNavigation.commitID(options:options,highlighted:"removed",controlEnabled:true))
        XCTAssertEqual(FlatChoiceNavigation.commitID(options:options,highlighted:"teal",controlEnabled:true),"teal")
    }
}
