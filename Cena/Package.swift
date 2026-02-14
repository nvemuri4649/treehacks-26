// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Cena",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "Cena",
            targets: ["Cena"]
        )
    ],
    targets: [
        .executableTarget(
            name: "Cena",
            path: "Cena"
        )
    ]
)
