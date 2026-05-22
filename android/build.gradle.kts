allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// Set default Java version for all subprojects to suppress obsolete Java 8 warnings
subprojects {
    tasks.withType<JavaCompile> {
        sourceCompatibility = "11"
        targetCompatibility = "11"
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
